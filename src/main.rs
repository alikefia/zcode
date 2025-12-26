use clap::{Parser, Subcommand};

mod llm;
mod lsp;

#[derive(Parser)]
#[command(name = "zc")]
#[command(about = "A locally running code assistant ", long_about = None)]
struct Cli {
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
async fn main() {
    tracing_subscriber::fmt().init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Lsp => lsp::run().await,
        Commands::Llm => {
            if let Err(e) = llm::run() {
                eprintln!("Error running LLM: {}", e);
                std::process::exit(1);
            }
        }
    }
}
