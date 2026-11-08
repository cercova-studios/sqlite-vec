use super::{Error, Result, Stage, Writer, identity};
use rusqlite::{Connection, LoadExtensionGuard, OptionalExtension, params};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Model {
    pub id: String,
    pub dimensions: usize,
}
impl Model {
    pub(super) fn validate(&self) -> Result<()> {
        identity(&self.id)?;
        if !(1..=4096).contains(&self.dimensions) {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct PreparedEmbedding {
    pub model: String,
    pub source: String,
    pub values: Vec<f32>,
}
impl PreparedEmbedding {
    pub(super) fn validate(&self, title: &str, model: Option<&Model>) -> Result<()> {
        let model = model.ok_or(Error::Invalid)?;
        if self.model != model.id || self.source != title {
            return Err(Error::Conflict);
        }
        if self.values.len() != model.dimensions
            || self.values.iter().any(|v| !v.is_finite() || v.abs() > 1e10)
        {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    pub(super) fn store(&self, db: &Connection, id: i64) -> Result<()> {
        let bytes: Vec<u8> = self.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.execute("INSERT INTO source_vectors VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET model=excluded.model,vector=excluded.vector",params![id,self.model,bytes])?;
        Ok(())
    }
}
#[derive(Debug, PartialEq)]
pub struct Source {
    pub title: String,
    pub model: String,
    pub values: Vec<f32>,
    pub source_revision: i64,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Snapshot {
    title: String,
    model: Option<String>,
    vector: Option<Vec<u8>>,
}
impl Snapshot {
    fn values(&self, model: &Model) -> Result<Vec<f32>> {
        let vector = self.vector.as_ref().ok_or(Error::Schema)?;
        if self.model.as_deref() != Some(model.id.as_str()) || vector.len() != model.dimensions * 4
        {
            return Err(Error::Schema);
        }
        let values: Vec<f32> = vector
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        if values.iter().any(|v| !v.is_finite() || v.abs() > 1e10) {
            return Err(Error::Schema);
        }
        Ok(values)
    }
}
pub(super) fn snapshot(db: &Connection, id: i64) -> Result<Option<Snapshot>> {
    Ok(db
        .query_row(
            "SELECT title,model,vector FROM tasks LEFT JOIN source_vectors USING(id) WHERE id=?",
            [id],
            |r| {
                Ok(Snapshot {
                    title: r.get(0)?,
                    model: r.get(1)?,
                    vector: r.get(2)?,
                })
            },
        )
        .optional()?)
}
pub(super) fn source_revision(db: &Connection, id: i64) -> Result<Option<i64>> {
    Ok(db
        .query_row(
            "SELECT source_revision FROM versions WHERE id=?",
            [id],
            |r| r.get(0),
        )
        .optional()?)
}
/// # Safety
/// The path must name trusted native code compatible with this SQLite host's extension ABI.
pub(super) unsafe fn load(db: &Connection, extension: &Path) -> Result<()> {
    // SAFETY: indexed constructors pass their documented trusted, ABI-compatible host input.
    unsafe {
        let _guard = LoadExtensionGuard::new(db)?;
        db.load_extension(extension, None::<&str>)?;
    }
    Ok(())
}
pub(super) fn initialize(db: &Connection, model: &Model) -> Result<()> {
    // Derived rows cannot constrain Session's transient canonical state before rebuilding.
    db.execute_batch(&format!("CREATE VIRTUAL TABLE lexical USING fts5(body);CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{}]);CREATE TABLE index_state(id INTEGER PRIMARY KEY,source_revision INTEGER NOT NULL,model TEXT NOT NULL);",model.dimensions))?;
    Ok(())
}
pub(super) fn validate_derived(
    db: &Connection,
    id: i64,
    value: Option<&Snapshot>,
    model: &Model,
) -> Result<()> {
    let lengths: (i64,i64,i64)=db.query_row("SELECT coalesce((SELECT length(CAST(body AS BLOB)) FROM lexical WHERE rowid=?1),0),coalesce((SELECT length(embedding) FROM vectors WHERE rowid=?1),0),coalesce((SELECT length(CAST(model AS BLOB)) FROM index_state WHERE id=?1),0)",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
    if !(0..=super::ROW_BYTES as i64).contains(&lengths.0)
        || !(0..=16384).contains(&lengths.1)
        || !(0..=128).contains(&lengths.2)
    {
        return Err(Error::Limit);
    }
    if value.is_some() && lengths.1 != model.dimensions as i64 * 4 {
        return Err(Error::Schema);
    }
    let text: Option<String> = db
        .query_row("SELECT body FROM lexical WHERE rowid=?", [id], |r| r.get(0))
        .optional()?;
    let vector: Option<Vec<u8>> = db
        .query_row("SELECT embedding FROM vectors WHERE rowid=?", [id], |r| {
            r.get(0)
        })
        .optional()?;
    let state: Option<(i64, String)> = db
        .query_row(
            "SELECT source_revision,model FROM index_state WHERE id=?",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    if let Some(value) = value {
        let expected: Vec<u8> = value
            .values(model)?
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        if text.as_deref() != Some(value.title.as_str())
            || vector.as_ref() != Some(&expected)
            || state != source_revision(db, id)?.map(|revision| (revision, model.id.clone()))
        {
            return Err(Error::Schema);
        }
    } else if text.is_some() || vector.is_some() || state.is_some() {
        return Err(Error::Schema);
    }
    Ok(())
}
impl Writer {
    pub(super) fn sync_derived(
        &self,
        db: &Connection,
        id: i64,
        value: Option<&Snapshot>,
        revision: i64,
        model: &Model,
        cancel: &AtomicBool,
    ) -> Result<()> {
        db.execute("DELETE FROM lexical WHERE rowid=?", [id])?;
        if let Some(value) = value {
            db.execute(
                "INSERT INTO lexical(rowid,body) VALUES(?,?)",
                params![id, value.title],
            )?;
        }
        self.checkpoint(Stage::Fts, db, cancel)?;
        db.execute("DELETE FROM vectors WHERE rowid=?", [id])?;
        if let Some(value) = value {
            let vector: Vec<u8> = value
                .values(model)?
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            db.execute(
                "INSERT INTO vectors(rowid,embedding) VALUES(?,?)",
                params![id, vector],
            )?;
        }
        self.checkpoint(Stage::Vector, db, cancel)?;
        db.execute("DELETE FROM index_state WHERE id=?", [id])?;
        if value.is_some() {
            db.execute(
                "INSERT INTO index_state VALUES(?,?,?)",
                params![id, revision, model.id],
            )?;
        }
        Ok(())
    }
    /// Read canonical embedding evidence and validate its logical derived row in one snapshot.
    pub fn source(&self, id: i64) -> Result<Option<Source>> {
        let model = self.model.as_ref().ok_or(Error::Invalid)?;
        let tx = self.connection()?.unchecked_transaction()?;
        self.check_schema(&tx)?;
        Self::row_bytes(&tx, &[id])?;
        let value = snapshot(&tx, id)?;
        validate_derived(&tx, id, value.as_ref(), model)?;
        let result = if let Some(value) = value {
            let values = value.values(model)?;
            Some(Source {
                title: value.title,
                model: model.id.clone(),
                values,
                source_revision: source_revision(&tx, id)?.ok_or(Error::Schema)?,
            })
        } else {
            None
        };
        tx.rollback()?;
        Ok(result)
    }
}
