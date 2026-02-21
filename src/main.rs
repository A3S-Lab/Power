use tracing_subscriber::EnvFilter;

use a3s_power::config::PowerConfig;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Handle --version and --help before anything else
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("a3s-power {VERSION}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("a3s-power {VERSION}");
        println!("Privacy-preserving local LLM inference server with TEE support.\n");
        println!("Usage: a3s-power [OPTIONS]\n");
        println!("Options:");
        println!("  -h, --help       Print help");
        println!("  -V, --version    Print version\n");
        println!("Configuration:");
        println!("  Config file: ~/.a3s/power/config.hcl");
        println!("  Env prefix:  A3S_POWER_*");
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = PowerConfig::load()?;
    a3s_power::server::start(config).await?;
    Ok(())
}
