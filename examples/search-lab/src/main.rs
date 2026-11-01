use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt, model::*, service::RequestContext};
use serde_json::{Value, json};
use sqlite_vec_search_lab::{Adapter, cli, error};
use std::{
    io,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

struct Server(
    Arc<Mutex<Adapter>>,
    Arc<rusqlite::InterruptHandle>,
    Vec<Tool>,
);

struct Cancellation {
    cancelled: Arc<AtomicBool>,
    active: Mutex<bool>,
    interrupt: Arc<rusqlite::InterruptHandle>,
}
impl Cancellation {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let active = self.active.lock().expect("cancellation gate");
        if *active {
            self.interrupt.interrupt();
        }
    }
}
struct CancelOnDrop(Arc<Cancellation>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            Implementation::new("sqlite-vec-search-lab", env!("CARGO_PKG_VERSION")),
        )
    }
    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            tools: self.2.clone(),
            ..Default::default()
        })
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arguments = request.arguments.unwrap_or_default();
        let result = if arguments.keys().any(|k| k != "corpus" && k != "args") {
            error("invalid_input", "request accepts only op, corpus and args")
        } else {
            let mut record = Value::Object(arguments);
            record["op"] = json!(request.name);
            let cancellation = Arc::new(Cancellation {
                cancelled: Arc::new(AtomicBool::new(false)),
                active: Mutex::new(false),
                interrupt: self.1.clone(),
            });
            let _guard = CancelOnDrop(cancellation.clone());
            let adapter = self.0.clone();
            let state = cancellation.clone();
            let mut work = tokio::task::spawn_blocking(move || {
                let mut adapter = adapter.lock().expect("serialized adapter");
                {
                    let mut active = state.active.lock().expect("cancellation gate");
                    if state.cancelled.load(Ordering::SeqCst) {
                        return error("cancelled", "request cancelled before execution");
                    }
                    *active = true;
                }
                let result = adapter.call_cancellable(&record, state.cancelled.clone());
                *state.active.lock().expect("cancellation gate") = false;
                result
            });
            tokio::select! {
                result = &mut work => result.map_err(|e|ErrorData::internal_error(e.to_string(), None))?,
                _ = context.ct.cancelled() => {
                    cancellation.cancel();
                    work.await.map_err(|e|ErrorData::internal_error(e.to_string(), None))?
                }
            }
        };
        let is_error = result["ok"] != true;
        let mut response = CallToolResult::structured(result);
        response.content = vec![ContentBlock::text("See structuredContent for the result.")];
        response.is_error = Some(is_error);
        Ok(response.into())
    }
}

async fn run() -> Result<i32, String> {
    let usage = "usage: sqlite-vec-search-lab --config PATH cli|mcp";
    let mut args = std::env::args().skip(1);
    let mut config = None;
    let mut transport = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{usage}");
                return Ok(0);
            }
            "--config" => {
                config = Some(PathBuf::from(
                    args.next().ok_or("--config requires a path")?,
                ))
            }
            "cli" | "mcp" if transport.is_none() => transport = Some(arg),
            _ => {
                if let Some(value) = arg.strip_prefix("--config=") {
                    config = Some(PathBuf::from(value));
                } else {
                    return Err(usage.into());
                }
            }
        }
    }
    let config = config.ok_or(usage)?;
    let transport = transport.ok_or(usage)?;
    let adapter = Adapter::open(&config).map_err(|e| e.message)?;
    if transport == "cli" {
        return cli(&adapter, io::stdin().lock(), io::stdout().lock()).map_err(|e| e.to_string());
    }
    let interrupt = Arc::new(adapter.interrupt_handle());
    let tools = adapter.tools();
    let service = Server(Arc::new(Mutex::new(adapter)), interrupt, tools)
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| e.to_string())?;
    service.waiting().await.map_err(|e| e.to_string())?;
    Ok(0)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let code = match run().await {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{}", error("startup_error", message));
            2
        }
    };
    std::process::exit(code);
}
