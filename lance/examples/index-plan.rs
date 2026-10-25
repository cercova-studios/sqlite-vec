//! Inspect the committed version used by the SQL ANN benchmark.
use lance::{Dataset, deps::arrow_array::Float32Array};
use lance::index::DatasetIndexExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: index-plan <dataset-uri> <version> <dimensions>".into());
    }
    let version: u64 = args[2].parse()?;
    let dimensions: usize = args[3].parse()?;
    tokio::runtime::Runtime::new()?.block_on(async {
        let ds = Dataset::open(&args[1]).await?.checkout_version(version).await?;
        let stats = ds.index_statistics("vector_idx").await?;
        let mut scan = ds.scan();
        scan.nearest("vector", &Float32Array::from(vec![0.0; dimensions]), 10)?;
        scan.prefilter(true);
        scan.minimum_nprobes(32).ef(128).refine(4);
        let plan = scan.explain_plan(false).await?;
        assert!(plan.contains("ANNSubIndex") && plan.contains("vector_idx"), "{plan}");
        println!("{stats}\n{plan}");
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}
