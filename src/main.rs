use anyhow::{Error as E, Result};
use clap::{Parser, Subcommand};
use std::fs::OpenOptions;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

mod llm;
mod lsp;
mod profiler;

#[derive(Parser)]
#[command(name = "zc")]
#[command(about = "A locally running code assistant ", long_about = None)]
struct Cli {
    /// Path to the trace log file
    #[arg(long, default_value = "/tmp/zc.log.jsonl")]
    trace_file: String,

    /// Trace filter (e.g., "debug", "info", "zc=trace")
    #[arg(long, default_value = "info")]
    filter: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the LSP server
    Lsp,
    /// Runs the llm for test complete
    Llm,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let trace_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cli.trace_file)
        .unwrap_or_else(|_| panic!("Failed to open trace file: {}", cli.trace_file));

    let filter = EnvFilter::try_new(&cli.filter)
        .unwrap_or_else(|_| panic!("Invalid filter: {}", cli.filter));

    tracing_subscriber::fmt()
        .json()
        .with_span_events(FmtSpan::FULL)
        .with_current_span(true)
        .with_env_filter(filter)
        .with_writer(std::sync::Mutex::new(trace_file))
        .init();

    match cli.command {
        Commands::Lsp => lsp::run().await.map_err(E::msg),
        Commands::Llm => llm::run(),
    }
}
