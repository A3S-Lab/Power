use tracing_subscriber::EnvFilter;

use a3s_power::config::PowerConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = PowerConfig::load()?;
    a3s_power::server::start(config).await?;
    Ok(())
}
