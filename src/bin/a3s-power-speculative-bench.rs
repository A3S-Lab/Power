use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use a3s_power::error::{PowerError, Result};
use a3s_power::speculative::benchmark::client::{run_benchmark, SpeculativeBenchmarkRunConfig};
use a3s_power::speculative::benchmark::{compare_reports, SpeculativeBenchmarkReport};
use a3s_power::speculative::SpeculativeStrategy;
use clap::{Args, Parser, Subcommand};
use reqwest::Url;
use zeroize::Zeroizing;

const MAX_PROMPT_BYTES: u64 = 1024 * 1024;
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "a3s-power-speculative-bench",
    version,
    about = "Capture and compare path-free speculative-decoding evidence through Power's SSE API"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run one explicit server mode and write its JSON report to stdout.
    Run(RunArgs),
    /// Compare an explicit-off baseline report with a speculative candidate.
    Compare(CompareArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Power server base URL. Plain HTTP is accepted only for loopback hosts.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    url: String,

    /// Exact registered model name exposed by GET /v1/models/:name.
    #[arg(long)]
    model: String,

    /// Expected lowercase SHA-256 of the registered GGUF artifact.
    #[arg(long)]
    model_sha256: String,

    /// Expected explicit server mode (for example, off or mtp).
    #[arg(long, value_parser = parse_mode)]
    mode: SpeculativeStrategy,

    /// Lowercase 40- or 64-character Power git revision used by the server build.
    #[arg(long)]
    power_commit: String,

    /// Stable operator-assigned name for the acceptance host and configuration.
    #[arg(long)]
    hardware_label: String,

    /// UTF-8 prompt file. Its content and path are never included in the report.
    #[arg(long)]
    prompt_file: PathBuf,

    /// Environment variable containing the bearer token; never pass a key directly.
    #[arg(long)]
    api_key_env: Option<String>,

    /// Fixed number of completion tokens required from every measured request.
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,

    /// Context size sent with every request.
    #[arg(long, default_value_t = 4096)]
    num_ctx: u32,

    /// Deterministic greedy-decoding seed.
    #[arg(long, default_value_t = 42)]
    seed: i64,

    /// Unmeasured requests made before sampling.
    #[arg(long, default_value_t = 1)]
    warmup_runs: u32,

    /// Number of measured requests.
    #[arg(long, default_value_t = 5)]
    samples: usize,

    /// Required median server-side decode throughput for this report.
    #[arg(long)]
    min_tokens_per_second: f64,

    /// Per-request HTTP timeout, including model autoload time.
    #[arg(long, default_value_t = 900)]
    timeout_secs: u64,
}

#[derive(Debug, Args)]
struct CompareArgs {
    /// JSON report captured with explicit spec_mode=off.
    baseline: PathBuf,
    /// JSON report captured with an explicit speculative strategy.
    candidate: PathBuf,
}

#[tokio::main]
async fn main() {
    if let Err(error) = execute(Cli::parse()).await {
        eprintln!("speculative benchmark failed: {error}");
        std::process::exit(2);
    }
}

async fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Run(args) => execute_run(args).await,
        Command::Compare(args) => execute_compare(args),
    }
}

async fn execute_run(args: RunArgs) -> Result<()> {
    let prompt = read_utf8_secret(&args.prompt_file, MAX_PROMPT_BYTES, "benchmark prompt")?;
    let api_key = args.api_key_env.as_deref().map(read_api_key).transpose()?;
    let base_url = Url::parse(&args.url)
        .map_err(|error| invalid(format!("invalid benchmark URL: {error}")))?;
    let config = SpeculativeBenchmarkRunConfig {
        base_url,
        api_key,
        model: args.model,
        expected_model_sha256: args.model_sha256,
        mode: args.mode,
        power_commit: args.power_commit,
        hardware_label: args.hardware_label,
        prompt,
        max_tokens: args.max_tokens,
        num_ctx: args.num_ctx,
        seed: args.seed,
        warmup_runs: args.warmup_runs,
        samples: args.samples,
        min_required_tokens_per_second: args.min_tokens_per_second,
        timeout: Duration::from_secs(args.timeout_secs),
    };
    let report = run_benchmark(&config).await?;
    write_json(&report)?;
    if report.threshold_passed {
        Ok(())
    } else {
        Err(invalid(format!(
            "median decode throughput {:.3} token/s did not reach the required {:.3} token/s",
            report.median_decode_tokens_per_second, report.min_required_tokens_per_second
        )))
    }
}

fn execute_compare(args: CompareArgs) -> Result<()> {
    let baseline = read_report(&args.baseline)?;
    let candidate = read_report(&args.candidate)?;
    let comparison = compare_reports(&baseline, &candidate)?;
    write_json(&comparison)?;
    if comparison.passed {
        Ok(())
    } else {
        Err(invalid(
            "speculative benchmark comparison failed output parity or candidate throughput",
        ))
    }
}

fn read_report(path: &Path) -> Result<SpeculativeBenchmarkReport> {
    let bytes = read_bounded_regular_file(path, MAX_REPORT_BYTES, "benchmark report")?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn read_utf8_secret(path: &Path, maximum: u64, label: &str) -> Result<Zeroizing<String>> {
    let bytes = Zeroizing::new(read_bounded_regular_file(path, maximum, label)?);
    let value = std::str::from_utf8(bytes.as_slice())
        .map_err(|_| invalid(format!("{label} must contain valid UTF-8")))?;
    Ok(Zeroizing::new(value.to_string()))
}

fn read_bounded_regular_file(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > maximum
    {
        return Err(invalid(format!(
            "{label} must be a non-empty regular non-symlink file no larger than {maximum} bytes"
        )));
    }
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() || bytes.len() as u64 > maximum {
        return Err(invalid(format!("{label} changed while it was being read")));
    }
    Ok(bytes)
}

fn read_api_key(variable: &str) -> Result<Zeroizing<String>> {
    if variable.is_empty()
        || variable.len() > 256
        || variable.contains('=')
        || variable.chars().any(char::is_control)
    {
        return Err(invalid("API key environment variable name is invalid"));
    }
    let value = Zeroizing::new(std::env::var(variable).map_err(|_| {
        invalid(format!(
            "API key environment variable '{variable}' is not set"
        ))
    })?);
    if value.is_empty() || value.len() > 64 * 1024 {
        return Err(invalid("API key must contain between 1 byte and 64 KiB"));
    }
    Ok(value)
}

fn parse_mode(value: &str) -> std::result::Result<SpeculativeStrategy, String> {
    let mode = SpeculativeStrategy::parse(value)
        .ok_or_else(|| format!("unsupported speculative strategy '{value}'"))?;
    if mode == SpeculativeStrategy::Auto {
        Err("benchmark mode must be explicit, not auto".to_string())
    } else {
        Ok(mode)
    }
}

fn write_json(value: &impl serde::Serialize) -> Result<()> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn invalid(message: impl Into<String>) -> PowerError {
    PowerError::InvalidRequest(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_arguments_require_an_explicit_mode_and_threshold() {
        let error = Cli::try_parse_from([
            "bench",
            "run",
            "--model",
            "qwen",
            "--model-sha256",
            &"a".repeat(64),
            "--mode",
            "auto",
            "--power-commit",
            &"b".repeat(40),
            "--hardware-label",
            "host",
            "--prompt-file",
            "prompt.txt",
            "--min-tokens-per-second",
            "100",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("explicit"));
    }

    #[test]
    fn mode_parser_normalizes_supported_aliases() {
        assert_eq!(
            parse_mode("multi-token-prediction").unwrap(),
            SpeculativeStrategy::Mtp
        );
        assert_eq!(parse_mode("none").unwrap(), SpeculativeStrategy::Off);
    }

    #[test]
    fn bounded_reader_rejects_empty_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty.json");
        std::fs::write(&path, []).unwrap();
        assert!(read_bounded_regular_file(&path, 32, "test file").is_err());
    }
}
