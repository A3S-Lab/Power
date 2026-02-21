//! a3s-power-verify — CLI tool for verifying TEE attestation reports.
//!
//! Fetches an attestation report from a running a3s-power server (or reads
//! from a JSON file) and verifies nonce binding, model hash binding, and
//! platform measurement.
//!
//! # Usage
//!
//! ```text
//! # Verify against a live server
//! a3s-power-verify --url http://localhost:11434 \
//!     --nonce deadbeef \
//!     --model-hash <sha256-hex> \
//!     --expected-measurement <hex>
//!
//! # Verify a saved report file
//! a3s-power-verify --file report.json \
//!     --nonce deadbeef \
//!     --model-hash <sha256-hex>
//! ```

use std::process;

use a3s_power::tee::attestation::AttestationReport;
use a3s_power::verify::{verify_report, VerifyOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args[1..]) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(args: &[String]) -> anyhow::Result<()> {
    let opts = parse_args(args)?;

    // Load the attestation report
    let report = load_report(&opts)?;

    // Build verify options
    let nonce = opts
        .nonce
        .as_deref()
        .map(hex::decode)
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid --nonce hex: {e}"))?;

    let model_hash = opts
        .model_hash
        .as_deref()
        .map(hex::decode)
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid --model-hash hex: {e}"))?;

    let expected_measurement = opts
        .expected_measurement
        .as_deref()
        .map(hex::decode)
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid --expected-measurement hex: {e}"))?;

    let verify_opts = VerifyOptions {
        nonce,
        expected_model_hash: model_hash,
        expected_measurement,
        hardware_verifier: None,
    };

    let result = verify_report(&report, &verify_opts)
        .map_err(|e| anyhow::anyhow!("verification failed: {e}"))?;

    // Print results
    println!("TEE type:    {}", report.tee_type);
    println!("Timestamp:   {}", report.timestamp);
    println!(
        "Nonce:       {}",
        if result.nonce_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "Model hash:  {}",
        if result.model_hash_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "Measurement: {}",
        if result.measurement_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "HW signature:{}",
        if result.hardware_verified {
            "✓ verified"
        } else {
            "— skipped (offline mode)"
        }
    );
    println!("\nAttestation OK");
    Ok(())
}

// ============================================================================
// Argument parsing (no external deps — keeps binary minimal)
// ============================================================================

struct CliOpts {
    /// URL of a running a3s-power server (e.g. http://localhost:11434)
    url: Option<String>,
    /// Path to a JSON file containing an AttestationReport
    file: Option<String>,
    /// Model name to include in the attestation request (?model=<name>)
    model: Option<String>,
    /// Client nonce (hex-encoded)
    nonce: Option<String>,
    /// Expected model SHA-256 hash (hex-encoded, 32 bytes)
    model_hash: Option<String>,
    /// Expected platform measurement (hex-encoded)
    expected_measurement: Option<String>,
}

fn parse_args(args: &[String]) -> anyhow::Result<CliOpts> {
    let mut opts = CliOpts {
        url: None,
        file: None,
        model: None,
        nonce: None,
        model_hash: None,
        expected_measurement: None,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--url" => {
                opts.url = Some(next_arg(args, &mut i, "--url")?);
            }
            "--file" => {
                opts.file = Some(next_arg(args, &mut i, "--file")?);
            }
            "--model" => {
                opts.model = Some(next_arg(args, &mut i, "--model")?);
            }
            "--nonce" => {
                opts.nonce = Some(next_arg(args, &mut i, "--nonce")?);
            }
            "--model-hash" => {
                opts.model_hash = Some(next_arg(args, &mut i, "--model-hash")?);
            }
            "--expected-measurement" => {
                opts.expected_measurement = Some(next_arg(args, &mut i, "--expected-measurement")?);
            }
            other => {
                return Err(anyhow::anyhow!("unknown argument: {other}"));
            }
        }
        i += 1;
    }

    if opts.url.is_none() && opts.file.is_none() {
        return Err(anyhow::anyhow!(
            "one of --url or --file is required. Run with --help for usage."
        ));
    }

    Ok(opts)
}

fn next_arg(args: &[String], i: &mut usize, flag: &str) -> anyhow::Result<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn print_help() {
    println!(
        r#"a3s-power-verify — verify TEE attestation reports from a3s-power

USAGE:
    a3s-power-verify [OPTIONS]

OPTIONS:
    --url <URL>                    Fetch report from a live server (e.g. http://localhost:11434)
    --file <PATH>                  Read report from a JSON file
    --model <NAME>                 Model name to bind into the attestation request
    --nonce <HEX>                  Client nonce to verify (hex-encoded)
    --model-hash <HEX>             Expected model SHA-256 hash (hex-encoded, 32 bytes)
    --expected-measurement <HEX>   Expected platform measurement (hex-encoded)
    --help                         Show this help

EXAMPLES:
    # Verify against a live server with nonce and model hash
    a3s-power-verify --url http://localhost:11434 \
        --model llama3 \
        --nonce deadbeef01234567 \
        --model-hash <64-char-hex>

    # Verify a saved report file
    a3s-power-verify --file report.json --nonce deadbeef

    # Check measurement only
    a3s-power-verify --file report.json \
        --expected-measurement <96-char-hex>
"#
    );
}

// ============================================================================
// Report loading
// ============================================================================

fn load_report(opts: &CliOpts) -> anyhow::Result<AttestationReport> {
    if let Some(ref path) = opts.file {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read {path}: {e}"))?;
        let report: AttestationReport = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("failed to parse report JSON: {e}"))?;
        return Ok(report);
    }

    if let Some(ref base_url) = opts.url {
        return fetch_report(base_url, opts.model.as_deref(), opts.nonce.as_deref());
    }

    unreachable!("parse_args ensures url or file is set")
}

fn fetch_report(
    base_url: &str,
    model: Option<&str>,
    nonce: Option<&str>,
) -> anyhow::Result<AttestationReport> {
    let mut url = format!("{base_url}/v1/attestation");
    let mut params: Vec<String> = Vec::new();

    if let Some(n) = nonce {
        params.push(format!("nonce={n}"));
    }
    if let Some(m) = model {
        params.push(format!("model={m}"));
    }
    if !params.is_empty() {
        url = format!("{url}?{}", params.join("&"));
    }

    eprintln!("Fetching attestation report from {url}");

    // Use std blocking HTTP via ureq (already in scope via anyhow chain)
    // We avoid adding tokio runtime here to keep the binary minimal.
    let response = ureq::get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;

    let body = response
        .into_string()
        .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;

    let report: AttestationReport = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("failed to parse attestation response: {e}"))?;

    Ok(report)
}
