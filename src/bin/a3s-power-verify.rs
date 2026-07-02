//! a3s-power-verify — CLI tool for verifying TEE attestation reports.
//!
//! Fetches an attestation report from a running a3s-power server (or reads
//! from a JSON file) and verifies hardware signatures by default, plus nonce
//! binding, model hash binding, and platform measurement when expected values
//! are supplied.
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
#[cfg(feature = "hw-verify")]
use std::time::Duration;

use a3s_power::api::prompt_policy::canonical_gpu_execution_digest;
use a3s_power::api::receipt::{AttestationReceipt, ReceiptRequestType};
use a3s_power::api::types::{ChatCompletionRequest, CompletionRequest};
use a3s_power::config::GpuConfig;
use a3s_power::tee::attestation::AttestationReport;
use a3s_power::verify::{
    verify_receipt_against_attestation, verify_receipt_matches_chat_request,
    verify_receipt_matches_completion_request, verify_receipt_policy, verify_report_with_policy,
    ExpectedGpuDevices, ExpectedGpuEvidence, ExpectedReceipt, HardwareVerifier, VerificationPolicy,
    VerifyOptions,
};
#[cfg(feature = "hw-verify")]
use a3s_power::verify::{SevSnpVerifier, TdxVerifier};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args[1..]) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(args: &[String]) -> anyhow::Result<()> {
    let opts = parse_args(args)?;

    if opts.print_gpu_execution_digest {
        println!("{}", gpu_execution_digest_from_opts(&opts)?);
        return Ok(());
    }

    if !opts.allow_offline && opts.expected_measurement.is_none() {
        anyhow::bail!(
            "strict verification requires --expected-measurement; use --allow-offline only for development/offline checks"
        );
    }

    if (opts.gpu_confidential
        || opts.require_gpu_evidence
        || opts.require_gpu_device_claims
        || opts.has_gpu_evidence_pins()
        || opts.has_gpu_device_pins())
        && opts.nonce.is_none()
    {
        anyhow::bail!(
            "--require-gpu-evidence, --require-gpu-device-claims, and GPU evidence/device pins require --nonce so the verifier can check GPU evidence freshness"
        );
    }
    if opts.gpu_confidential
        && opts
            .nonce
            .as_deref()
            .is_some_and(|nonce| nonce.trim().len() != 64)
    {
        anyhow::bail!(
            "--gpu-confidential requires --nonce to be a 32-byte value encoded as 64 hex characters"
        );
    }
    if opts.gpu_confidential && opts.gpu_verdict_digest.is_none() {
        anyhow::bail!(
            "--gpu-confidential requires --gpu-verdict-digest so the NVIDIA NRAS verdict is pinned"
        );
    }
    if opts.gpu_confidential && !opts.has_gpu_confidential_evidence_pins() {
        anyhow::bail!(
            "--gpu-confidential requires --gpu-provider, --gpu-evidence-format, --gpu-verdict-format, and a nonzero --gpu-evidence-count"
        );
    }
    if opts.gpu_confidential && opts.gpu_execution_digest.is_none() {
        anyhow::bail!("--gpu-confidential requires --gpu-execution-digest");
    }
    if opts.gpu_confidential && !opts.has_gpu_confidential_device_pins() {
        anyhow::bail!(
            "--gpu-confidential requires --gpu-claims-version plus an exact --gpu-ueid set, or --gpu-claims-version plus --gpu-count and at least one of --gpu-hwmodel, --gpu-driver-version, or --gpu-firmware-version; when --nvswitch-count is nonzero it also requires --nvswitch-claims-version plus an exact --nvswitch-ueid set, or --nvswitch-claims-version plus at least one of --nvswitch-hwmodel or --nvswitch-firmware-version"
        );
    }
    if opts.receipt_file.is_none()
        && (opts.receipt_digest.is_some()
            || opts.effective_prompt_digest.is_some()
            || opts.has_receipt_request_file()
            || opts.has_receipt_policy_pins())
    {
        anyhow::bail!(
            "--receipt-digest, --effective-prompt-digest, receipt request files, and receipt policy pins require --receipt-file"
        );
    }
    if opts.receipt_chat_request_file.is_some() && opts.receipt_completion_request_file.is_some() {
        anyhow::bail!(
            "--receipt-chat-request-file conflicts with --receipt-completion-request-file"
        );
    }
    if opts.effective_prompt_absent
        && (opts.effective_prompt_digest.is_some()
            || opts
                .effective_prompt_backend
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || opts
                .effective_prompt_kind
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty()))
    {
        anyhow::bail!(
            "--require-effective-prompt-absent conflicts with --effective-prompt-digest, --effective-prompt-backend, and --effective-prompt-kind"
        );
    }

    // Load the attestation report
    let report = load_report(&opts)?;
    let receipt = load_receipt(&opts)?;
    let receipt_request = load_receipt_request(&opts)?;

    // Build verify options
    let nonce = opts
        .nonce
        .as_deref()
        .map(hex::decode)
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid --nonce hex: {e}"))?;

    let model_hash = decode_optional_hex_arg("--model-hash", opts.model_hash.as_deref())?;

    let expected_measurement = opts
        .expected_measurement
        .as_deref()
        .map(hex::decode)
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid --expected-measurement hex: {e}"))?;

    let expected_gpu_evidence_digest =
        decode_optional_hex_arg("--gpu-evidence-digest", opts.gpu_evidence_digest.as_deref())?;
    let expected_gpu_verdict_digest =
        decode_optional_hex_arg("--gpu-verdict-digest", opts.gpu_verdict_digest.as_deref())?;
    let expected_gpu_evidence = opts.expected_gpu_evidence();
    let expected_gpu_devices = opts.expected_gpu_devices();
    let expected_chat_template_digest = decode_optional_hex_arg(
        "--chat-template-digest",
        opts.chat_template_digest.as_deref(),
    )?;
    let expected_decoding_parameters_digest = decode_optional_hex_arg(
        "--decoding-policy-digest",
        opts.decoding_policy_digest.as_deref(),
    )?;
    let expected_gpu_execution_digest = decode_optional_hex_arg(
        "--gpu-execution-digest",
        opts.gpu_execution_digest.as_deref(),
    )?;
    let expected_receipt_digest =
        decode_optional_hex_arg("--receipt-digest", opts.receipt_digest.as_deref())?;
    let expected_effective_prompt_digest = decode_optional_hex_arg(
        "--effective-prompt-digest",
        opts.effective_prompt_digest.as_deref(),
    )?;
    let expected_receipt_input_digest = decode_optional_hex_arg(
        "--receipt-input-digest",
        opts.receipt_input_digest.as_deref(),
    )?;
    let expected_receipt_decoding_parameters_digest = decode_optional_hex_arg(
        "--receipt-decoding-parameters-digest",
        opts.receipt_decoding_parameters_digest.as_deref(),
    )?;
    let expected_receipt_stream_options_digest = decode_optional_hex_arg(
        "--receipt-stream-options-digest",
        opts.receipt_stream_options_digest.as_deref(),
    )?;
    let expected_receipt_stop_tokens_digest = decode_optional_hex_arg(
        "--receipt-stop-tokens-digest",
        opts.receipt_stop_tokens_digest.as_deref(),
    )?;
    let expected_receipt_response_format_digest = decode_optional_hex_arg(
        "--receipt-response-format-digest",
        opts.receipt_response_format_digest.as_deref(),
    )?;
    let expected_receipt_tools_digest = decode_optional_hex_arg(
        "--receipt-tools-digest",
        opts.receipt_tools_digest.as_deref(),
    )?;
    let expected_receipt_tool_choice_digest = decode_optional_hex_arg(
        "--receipt-tool-choice-digest",
        opts.receipt_tool_choice_digest.as_deref(),
    )?;
    let mut expected_receipt = opts.expected_receipt(
        expected_receipt_input_digest,
        expected_receipt_decoding_parameters_digest,
        expected_receipt_stream_options_digest,
        expected_receipt_stop_tokens_digest,
        expected_receipt_response_format_digest,
        expected_receipt_tools_digest,
        expected_receipt_tool_choice_digest,
        expected_effective_prompt_digest.clone(),
    );
    apply_image_receipt_prompt_absence_default(&mut expected_receipt, receipt_request.as_ref());

    let hardware_verifier = if opts.allow_offline {
        None
    } else {
        Some(default_hardware_verifier(&report, &opts)?)
    };

    let verify_opts = VerifyOptions {
        nonce,
        expected_model_hash: model_hash,
        expected_measurement,
        expected_gpu_evidence_digest,
        expected_gpu_verdict_digest,
        expected_gpu_evidence,
        expected_gpu_devices,
        expected_chat_template_digest,
        expected_decoding_parameters_digest,
        expected_gpu_execution_digest,
        hardware_verifier: hardware_verifier.as_deref(),
    };

    let policy = if opts.allow_offline {
        VerificationPolicy::permissive()
    } else {
        VerificationPolicy::strict()
    };
    let policy = if opts.gpu_confidential {
        policy.require_gpu_confidential()
    } else {
        policy
    };
    let policy = if opts.require_gpu_evidence {
        policy.require_gpu_evidence()
    } else {
        policy
    };
    let policy = if opts.require_gpu_device_claims || opts.has_gpu_device_pins() {
        policy.require_gpu_device_claims()
    } else {
        policy
    };
    let policy = if opts.require_runtime_policy {
        policy.require_runtime_policy()
    } else {
        policy
    };

    let result = verify_report_with_policy(&report, &verify_opts, policy)
        .map_err(|e| anyhow::anyhow!("verification failed: {e}"))?;
    if let Some(receipt) = receipt.as_ref() {
        verify_receipt_against_attestation(
            &report,
            receipt,
            expected_receipt_digest.as_deref(),
            expected_effective_prompt_digest.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("receipt verification failed: {e}"))?;
        if !expected_receipt.is_empty() {
            verify_receipt_policy(receipt, &expected_receipt)
                .map_err(|e| anyhow::anyhow!("receipt verification failed: {e}"))?;
        }
        if let Some(receipt_request) = receipt_request.as_ref() {
            match receipt_request {
                ReceiptRequest::Chat(request) => {
                    verify_receipt_matches_chat_request(receipt, request)
                }
                ReceiptRequest::Completion(request) => {
                    verify_receipt_matches_completion_request(receipt, request)
                }
            }
            .map_err(|e| anyhow::anyhow!("receipt request verification failed: {e}"))?;
        }
    }

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
        } else if opts.allow_offline {
            "— skipped (explicit offline mode)"
        } else {
            "— skipped"
        }
    );
    println!(
        "GPU evidence:{}",
        if result.gpu_evidence_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "GPU devices: {}",
        if result.gpu_device_claims_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "Runtime:     {}",
        if result.runtime_policy_verified {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!(
        "Receipt:     {}",
        if receipt.is_some() {
            "✓ verified"
        } else {
            "— skipped"
        }
    );
    println!("\nAttestation OK");
    Ok(())
}

// ============================================================================
// Argument parsing (no external deps — keeps binary minimal)
// ============================================================================

#[derive(Debug)]
struct CliOpts {
    /// URL of a running a3s-power server (e.g. http://localhost:11434)
    url: Option<String>,
    /// Path to a JSON file containing an AttestationReport
    file: Option<String>,
    /// Path to a JSON file containing an AttestationReceipt.
    receipt_file: Option<String>,
    /// Path to the original ChatCompletionRequest JSON covered by the receipt.
    receipt_chat_request_file: Option<String>,
    /// Path to the original CompletionRequest JSON covered by the receipt.
    receipt_completion_request_file: Option<String>,
    /// Model name to include in the attestation request (?model=<name>)
    model: Option<String>,
    /// Client nonce (hex-encoded)
    nonce: Option<String>,
    /// Expected model SHA-256 hash (hex-encoded, 32 bytes)
    model_hash: Option<String>,
    /// Expected platform measurement (hex-encoded)
    expected_measurement: Option<String>,
    /// Expected SHA-256 digest of NVIDIA GPU CC evidence bytes (hex-encoded)
    gpu_evidence_digest: Option<String>,
    /// Expected SHA-256 digest of NVIDIA NRAS verdict bytes (hex-encoded)
    gpu_verdict_digest: Option<String>,
    /// Expected GPU evidence provider label.
    gpu_provider: Option<String>,
    /// Expected GPU evidence byte-format label.
    gpu_evidence_format: Option<String>,
    /// Expected GPU verdict byte-format label.
    gpu_verdict_format: Option<String>,
    /// Expected number of GPU evidence entries.
    gpu_evidence_count: Option<u32>,
    /// Require the production NVIDIA GPU confidential-computing verifier profile.
    gpu_confidential: bool,
    /// Require a v2 GPU evidence claim and expected GPU digest verification.
    require_gpu_evidence: bool,
    /// Require structured NVIDIA GPU/NVSwitch identity and freshness claims.
    require_gpu_device_claims: bool,
    /// Expected number of NVIDIA GPU device claims.
    gpu_count: Option<u32>,
    /// Expected number of NVIDIA NVSwitch device claims.
    nvswitch_count: Option<u32>,
    /// Exact expected NVIDIA NVSwitch UEID set.
    nvswitch_ueids: Vec<String>,
    /// Allowed NVIDIA NVSwitch OEM IDs.
    nvswitch_oemids: Vec<String>,
    /// Allowed NVIDIA NVSwitch claims schema versions.
    nvswitch_claims_versions: Vec<String>,
    /// Allowed NVIDIA NVSwitch hardware model strings.
    nvswitch_hwmodels: Vec<String>,
    /// Allowed NVIDIA NVSwitch firmware/BIOS versions.
    nvswitch_firmware_versions: Vec<String>,
    /// Exact expected NVIDIA GPU UEID set.
    gpu_ueids: Vec<String>,
    /// Allowed NVIDIA GPU OEM IDs.
    gpu_oemids: Vec<String>,
    /// Allowed NVIDIA GPU claims schema versions.
    gpu_claims_versions: Vec<String>,
    /// Allowed NVIDIA GPU hardware model strings.
    gpu_hwmodels: Vec<String>,
    /// Allowed NVIDIA GPU driver versions.
    gpu_driver_versions: Vec<String>,
    /// Allowed NVIDIA GPU firmware/VBIOS versions.
    gpu_firmware_versions: Vec<String>,
    /// Expected SHA-256 digest of the effective chat template string.
    chat_template_digest: Option<String>,
    /// Expected SHA-256 digest of canonical default decoding parameters.
    decoding_policy_digest: Option<String>,
    /// Expected SHA-256 digest of canonical GPU execution parameters.
    gpu_execution_digest: Option<String>,
    /// Print the canonical GPU execution digest for supplied GPU parameters.
    print_gpu_execution_digest: bool,
    /// GPU layer offload value used by --print-gpu-execution-digest.
    digest_gpu_layers: Option<i32>,
    /// Main GPU index used by --print-gpu-execution-digest.
    digest_main_gpu: i32,
    /// Tensor split used by --print-gpu-execution-digest.
    digest_tensor_split: Vec<f32>,
    /// Require a v2 runtime policy claim and expected runtime digest verification.
    require_runtime_policy: bool,
    /// Expected SHA-256 digest of the canonical attestation receipt.
    receipt_digest: Option<String>,
    /// Expected model name in the receipt.
    receipt_model: Option<String>,
    /// Expected request type in the receipt.
    receipt_request_type: Option<ReceiptRequestType>,
    /// Expected SHA-256 digest of the receipt prompt-bearing input.
    receipt_input_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt decoding parameter map.
    receipt_decoding_parameters_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt stream-options JSON value.
    receipt_stream_options_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt stop-token JSON value.
    receipt_stop_tokens_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt response-format JSON value.
    receipt_response_format_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt tools JSON value.
    receipt_tools_digest: Option<String>,
    /// Expected SHA-256 digest of the receipt tool-choice JSON value.
    receipt_tool_choice_digest: Option<String>,
    /// Expected SHA-256 digest of the backend effective prompt representation.
    effective_prompt_digest: Option<String>,
    /// Require the receipt to omit effective_prompt.
    effective_prompt_absent: bool,
    /// Expected backend label for the receipt effective prompt digest.
    effective_prompt_backend: Option<String>,
    /// Expected semantic kind for the receipt effective prompt digest.
    effective_prompt_kind: Option<String>,
    /// Explicitly allow offline/development verification without hardware signatures.
    allow_offline: bool,
    /// Override the AMD KDS / Intel PCS certificate cache TTL in seconds.
    hw_cert_cache_ttl_secs: Option<u64>,
}

enum ReceiptRequest {
    Chat(ChatCompletionRequest),
    Completion(CompletionRequest),
}

impl CliOpts {
    fn has_gpu_evidence_pins(&self) -> bool {
        self.gpu_provider
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
            || self
                .gpu_evidence_format
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .gpu_verdict_format
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self.gpu_evidence_count.is_some()
    }

    fn has_gpu_confidential_evidence_pins(&self) -> bool {
        self.gpu_provider
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
            && self
                .gpu_evidence_format
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            && self
                .gpu_verdict_format
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            && self
                .gpu_evidence_count
                .is_some_and(|evidence_count| evidence_count > 0)
    }

    fn expected_gpu_evidence(&self) -> Option<ExpectedGpuEvidence> {
        self.has_gpu_evidence_pins().then(|| ExpectedGpuEvidence {
            provider: self.gpu_provider.clone(),
            evidence_format: self.gpu_evidence_format.clone(),
            verdict_format: self.gpu_verdict_format.clone(),
            evidence_count: self.gpu_evidence_count,
        })
    }

    fn has_gpu_device_pins(&self) -> bool {
        self.gpu_count.is_some()
            || self.nvswitch_count.is_some()
            || !self.gpu_ueids.is_empty()
            || !self.gpu_oemids.is_empty()
            || !self.gpu_claims_versions.is_empty()
            || !self.gpu_hwmodels.is_empty()
            || !self.gpu_driver_versions.is_empty()
            || !self.gpu_firmware_versions.is_empty()
            || !self.nvswitch_ueids.is_empty()
            || !self.nvswitch_oemids.is_empty()
            || !self.nvswitch_claims_versions.is_empty()
            || !self.nvswitch_hwmodels.is_empty()
            || !self.nvswitch_firmware_versions.is_empty()
    }

    fn has_gpu_confidential_device_pins(&self) -> bool {
        let has_identity_version_pin = !self.gpu_hwmodels.is_empty()
            || !self.gpu_driver_versions.is_empty()
            || !self.gpu_firmware_versions.is_empty();
        let has_gpu_pins = !self.gpu_claims_versions.is_empty()
            && (!self.gpu_ueids.is_empty()
                || self
                    .gpu_count
                    .is_some_and(|gpu_count| gpu_count > 0 && has_identity_version_pin));

        let nvswitch_count = self.nvswitch_count.unwrap_or_default();
        let has_nvswitch_identity_version_pin =
            !self.nvswitch_hwmodels.is_empty() || !self.nvswitch_firmware_versions.is_empty();
        let has_required_nvswitch_pins = nvswitch_count == 0
            || (!self.nvswitch_claims_versions.is_empty()
                && (!self.nvswitch_ueids.is_empty() || has_nvswitch_identity_version_pin));

        has_gpu_pins && has_required_nvswitch_pins
    }

    fn expected_gpu_devices(&self) -> Option<ExpectedGpuDevices> {
        self.has_gpu_device_pins().then(|| ExpectedGpuDevices {
            gpu_count: self.gpu_count,
            nvswitch_count: self.nvswitch_count,
            gpu_ueids: self.gpu_ueids.clone(),
            oemids: self.gpu_oemids.clone(),
            claims_versions: self.gpu_claims_versions.clone(),
            hwmodels: self.gpu_hwmodels.clone(),
            driver_versions: self.gpu_driver_versions.clone(),
            firmware_versions: self.gpu_firmware_versions.clone(),
            nvswitch_ueids: self.nvswitch_ueids.clone(),
            nvswitch_oemids: self.nvswitch_oemids.clone(),
            nvswitch_claims_versions: self.nvswitch_claims_versions.clone(),
            nvswitch_hwmodels: self.nvswitch_hwmodels.clone(),
            nvswitch_firmware_versions: self.nvswitch_firmware_versions.clone(),
        })
    }

    fn has_receipt_policy_pins(&self) -> bool {
        self.receipt_model
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
            || self.receipt_request_type.is_some()
            || self
                .receipt_input_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_decoding_parameters_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_stream_options_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_stop_tokens_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_response_format_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_tools_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .receipt_tool_choice_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .effective_prompt_digest
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self.effective_prompt_absent
            || self
                .effective_prompt_backend
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
            || self
                .effective_prompt_kind
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
    }

    fn has_receipt_request_file(&self) -> bool {
        self.receipt_chat_request_file.is_some() || self.receipt_completion_request_file.is_some()
    }

    fn expected_receipt(
        &self,
        input_digest: Option<Vec<u8>>,
        decoding_parameters_digest: Option<Vec<u8>>,
        stream_options_digest: Option<Vec<u8>>,
        stop_tokens_digest: Option<Vec<u8>>,
        response_format_digest: Option<Vec<u8>>,
        tools_digest: Option<Vec<u8>>,
        tool_choice_digest: Option<Vec<u8>>,
        effective_prompt_digest: Option<Vec<u8>>,
    ) -> ExpectedReceipt {
        ExpectedReceipt {
            model: self.receipt_model.clone(),
            request_type: self.receipt_request_type,
            input_digest,
            decoding_parameters_digest,
            stream_options_digest,
            stop_tokens_digest,
            response_format_digest,
            tools_digest,
            tool_choice_digest,
            effective_prompt_digest,
            effective_prompt_absent: self.effective_prompt_absent,
            effective_prompt_backend: normalized_optional_string_arg(
                self.effective_prompt_backend.as_deref(),
            ),
            effective_prompt_kind: normalized_optional_string_arg(
                self.effective_prompt_kind.as_deref(),
            ),
        }
    }
}

fn parse_args(args: &[String]) -> anyhow::Result<CliOpts> {
    let mut opts = CliOpts {
        url: None,
        file: None,
        receipt_file: None,
        receipt_chat_request_file: None,
        receipt_completion_request_file: None,
        model: None,
        nonce: None,
        model_hash: None,
        expected_measurement: None,
        gpu_evidence_digest: None,
        gpu_verdict_digest: None,
        gpu_provider: None,
        gpu_evidence_format: None,
        gpu_verdict_format: None,
        gpu_evidence_count: None,
        gpu_confidential: false,
        require_gpu_evidence: false,
        require_gpu_device_claims: false,
        gpu_count: None,
        nvswitch_count: None,
        nvswitch_ueids: Vec::new(),
        nvswitch_oemids: Vec::new(),
        nvswitch_claims_versions: Vec::new(),
        nvswitch_hwmodels: Vec::new(),
        nvswitch_firmware_versions: Vec::new(),
        gpu_ueids: Vec::new(),
        gpu_oemids: Vec::new(),
        gpu_claims_versions: Vec::new(),
        gpu_hwmodels: Vec::new(),
        gpu_driver_versions: Vec::new(),
        gpu_firmware_versions: Vec::new(),
        chat_template_digest: None,
        decoding_policy_digest: None,
        gpu_execution_digest: None,
        print_gpu_execution_digest: false,
        digest_gpu_layers: None,
        digest_main_gpu: 0,
        digest_tensor_split: Vec::new(),
        require_runtime_policy: false,
        receipt_digest: None,
        receipt_model: None,
        receipt_request_type: None,
        receipt_input_digest: None,
        receipt_decoding_parameters_digest: None,
        receipt_stream_options_digest: None,
        receipt_stop_tokens_digest: None,
        receipt_response_format_digest: None,
        receipt_tools_digest: None,
        receipt_tool_choice_digest: None,
        effective_prompt_digest: None,
        effective_prompt_absent: false,
        effective_prompt_backend: None,
        effective_prompt_kind: None,
        allow_offline: false,
        hw_cert_cache_ttl_secs: None,
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
            "--receipt-file" => {
                opts.receipt_file = Some(next_arg(args, &mut i, "--receipt-file")?);
            }
            "--receipt-chat-request-file" => {
                opts.receipt_chat_request_file =
                    Some(next_arg(args, &mut i, "--receipt-chat-request-file")?);
            }
            "--receipt-completion-request-file" => {
                opts.receipt_completion_request_file =
                    Some(next_arg(args, &mut i, "--receipt-completion-request-file")?);
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
            "--gpu-evidence-digest" => {
                opts.gpu_evidence_digest = Some(next_arg(args, &mut i, "--gpu-evidence-digest")?);
            }
            "--gpu-verdict-digest" => {
                opts.gpu_verdict_digest = Some(next_arg(args, &mut i, "--gpu-verdict-digest")?);
            }
            "--gpu-provider" => {
                opts.gpu_provider = Some(next_arg(args, &mut i, "--gpu-provider")?);
            }
            "--gpu-evidence-format" => {
                opts.gpu_evidence_format = Some(next_arg(args, &mut i, "--gpu-evidence-format")?);
            }
            "--gpu-verdict-format" => {
                opts.gpu_verdict_format = Some(next_arg(args, &mut i, "--gpu-verdict-format")?);
            }
            "--gpu-evidence-count" => {
                let value = next_arg(args, &mut i, "--gpu-evidence-count")?;
                let evidence_count = parse_u32_arg("--gpu-evidence-count", &value)?;
                if evidence_count == 0 {
                    anyhow::bail!("--gpu-evidence-count must be greater than zero");
                }
                opts.gpu_evidence_count = Some(evidence_count);
            }
            "--gpu-confidential" => {
                opts.gpu_confidential = true;
            }
            "--require-gpu-evidence" => {
                opts.require_gpu_evidence = true;
            }
            "--require-gpu-device-claims" => {
                opts.require_gpu_device_claims = true;
            }
            "--gpu-count" => {
                let value = next_arg(args, &mut i, "--gpu-count")?;
                let gpu_count = parse_u32_arg("--gpu-count", &value)?;
                if gpu_count == 0 {
                    anyhow::bail!("--gpu-count must be greater than zero");
                }
                opts.gpu_count = Some(gpu_count);
            }
            "--nvswitch-count" => {
                let value = next_arg(args, &mut i, "--nvswitch-count")?;
                opts.nvswitch_count = Some(parse_u32_arg("--nvswitch-count", &value)?);
            }
            "--nvswitch-ueid" => {
                opts.nvswitch_ueids.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--nvswitch-ueid",
                )?));
            }
            "--nvswitch-oemid" => {
                opts.nvswitch_oemids.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--nvswitch-oemid",
                )?));
            }
            "--nvswitch-claims-version" => {
                opts.nvswitch_claims_versions
                    .extend(split_csv_arg(&next_arg(
                        args,
                        &mut i,
                        "--nvswitch-claims-version",
                    )?));
            }
            "--nvswitch-hwmodel" => {
                opts.nvswitch_hwmodels.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--nvswitch-hwmodel",
                )?));
            }
            "--nvswitch-firmware-version" => {
                opts.nvswitch_firmware_versions
                    .extend(split_csv_arg(&next_arg(
                        args,
                        &mut i,
                        "--nvswitch-firmware-version",
                    )?));
            }
            "--gpu-ueid" => {
                opts.gpu_ueids
                    .extend(split_csv_arg(&next_arg(args, &mut i, "--gpu-ueid")?));
            }
            "--gpu-oemid" => {
                opts.gpu_oemids
                    .extend(split_csv_arg(&next_arg(args, &mut i, "--gpu-oemid")?));
            }
            "--gpu-claims-version" => {
                opts.gpu_claims_versions.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--gpu-claims-version",
                )?));
            }
            "--gpu-hwmodel" => {
                opts.gpu_hwmodels
                    .extend(split_csv_arg(&next_arg(args, &mut i, "--gpu-hwmodel")?));
            }
            "--gpu-driver-version" => {
                opts.gpu_driver_versions.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--gpu-driver-version",
                )?));
            }
            "--gpu-firmware-version" => {
                opts.gpu_firmware_versions.extend(split_csv_arg(&next_arg(
                    args,
                    &mut i,
                    "--gpu-firmware-version",
                )?));
            }
            "--chat-template-digest" => {
                opts.chat_template_digest = Some(next_arg(args, &mut i, "--chat-template-digest")?);
            }
            "--decoding-policy-digest" => {
                opts.decoding_policy_digest =
                    Some(next_arg(args, &mut i, "--decoding-policy-digest")?);
            }
            "--gpu-execution-digest" => {
                opts.gpu_execution_digest = Some(next_arg(args, &mut i, "--gpu-execution-digest")?);
            }
            "--print-gpu-execution-digest" => {
                opts.print_gpu_execution_digest = true;
            }
            "--gpu-layers" => {
                let value = next_arg(args, &mut i, "--gpu-layers")?;
                opts.digest_gpu_layers =
                    Some(value.parse::<i32>().map_err(|e| {
                        anyhow::anyhow!("--gpu-layers must be a signed integer: {e}")
                    })?);
            }
            "--main-gpu" => {
                let value = next_arg(args, &mut i, "--main-gpu")?;
                opts.digest_main_gpu = value
                    .parse::<i32>()
                    .map_err(|e| anyhow::anyhow!("--main-gpu must be a signed integer: {e}"))?;
            }
            "--tensor-split" => {
                let value = next_arg(args, &mut i, "--tensor-split")?;
                opts.digest_tensor_split = parse_tensor_split_arg(&value)?;
            }
            "--require-runtime-policy" => {
                opts.require_runtime_policy = true;
            }
            "--receipt-digest" => {
                opts.receipt_digest = Some(next_arg(args, &mut i, "--receipt-digest")?);
            }
            "--receipt-model" => {
                opts.receipt_model = Some(next_arg(args, &mut i, "--receipt-model")?);
            }
            "--receipt-request-type" => {
                let value = next_arg(args, &mut i, "--receipt-request-type")?;
                opts.receipt_request_type = Some(parse_receipt_request_type_arg(&value)?);
            }
            "--receipt-input-digest" => {
                opts.receipt_input_digest = Some(next_arg(args, &mut i, "--receipt-input-digest")?);
            }
            "--receipt-decoding-parameters-digest" => {
                opts.receipt_decoding_parameters_digest = Some(next_arg(
                    args,
                    &mut i,
                    "--receipt-decoding-parameters-digest",
                )?);
            }
            "--receipt-stream-options-digest" => {
                opts.receipt_stream_options_digest =
                    Some(next_arg(args, &mut i, "--receipt-stream-options-digest")?);
            }
            "--receipt-stop-tokens-digest" => {
                opts.receipt_stop_tokens_digest =
                    Some(next_arg(args, &mut i, "--receipt-stop-tokens-digest")?);
            }
            "--receipt-response-format-digest" => {
                opts.receipt_response_format_digest =
                    Some(next_arg(args, &mut i, "--receipt-response-format-digest")?);
            }
            "--receipt-tools-digest" => {
                opts.receipt_tools_digest = Some(next_arg(args, &mut i, "--receipt-tools-digest")?);
            }
            "--receipt-tool-choice-digest" => {
                opts.receipt_tool_choice_digest =
                    Some(next_arg(args, &mut i, "--receipt-tool-choice-digest")?);
            }
            "--effective-prompt-digest" => {
                opts.effective_prompt_digest =
                    Some(next_arg(args, &mut i, "--effective-prompt-digest")?);
            }
            "--require-effective-prompt-absent" => {
                opts.effective_prompt_absent = true;
            }
            "--effective-prompt-backend" => {
                opts.effective_prompt_backend =
                    Some(next_arg(args, &mut i, "--effective-prompt-backend")?);
            }
            "--effective-prompt-kind" => {
                opts.effective_prompt_kind =
                    Some(next_arg(args, &mut i, "--effective-prompt-kind")?);
            }
            "--allow-offline" => {
                opts.allow_offline = true;
            }
            "--hw-cert-cache-ttl-secs" => {
                opts.hw_cert_cache_ttl_secs = Some(parse_u64_arg(
                    "--hw-cert-cache-ttl-secs",
                    &next_arg(args, &mut i, "--hw-cert-cache-ttl-secs")?,
                )?);
            }
            other => {
                return Err(anyhow::anyhow!("unknown argument: {other}"));
            }
        }
        i += 1;
    }

    if !opts.print_gpu_execution_digest && opts.url.is_none() && opts.file.is_none() {
        return Err(anyhow::anyhow!(
            "one of --url, --file, or --print-gpu-execution-digest is required. Run with --help for usage."
        ));
    }

    Ok(opts)
}

#[cfg(feature = "hw-verify")]
fn default_hardware_verifier(
    report: &AttestationReport,
    opts: &CliOpts,
) -> anyhow::Result<Box<dyn HardwareVerifier>> {
    let ttl = opts.hw_cert_cache_ttl_secs.map(Duration::from_secs);
    match report.tee_type {
        a3s_power::tee::attestation::TeeType::SevSnp => Ok(ttl
            .map(SevSnpVerifier::with_ttl)
            .map(|verifier| Box::new(verifier) as Box<dyn HardwareVerifier>)
            .unwrap_or_else(|| Box::new(SevSnpVerifier::new()))),
        a3s_power::tee::attestation::TeeType::Tdx => Ok(ttl
            .map(TdxVerifier::with_ttl)
            .map(|verifier| Box::new(verifier) as Box<dyn HardwareVerifier>)
            .unwrap_or_else(|| Box::new(TdxVerifier::new()))),
        a3s_power::tee::attestation::TeeType::Simulated => Err(anyhow::anyhow!(
            "simulated TEE reports cannot be hardware-verified; use --allow-offline only for development"
        )),
        a3s_power::tee::attestation::TeeType::None => Err(anyhow::anyhow!(
            "tee_type=none cannot be hardware-verified"
        )),
    }
}

#[cfg(not(feature = "hw-verify"))]
fn default_hardware_verifier(
    _report: &AttestationReport,
    _opts: &CliOpts,
) -> anyhow::Result<Box<dyn HardwareVerifier>> {
    Err(anyhow::anyhow!(
        "hardware signature verification requires building a3s-power-verify with --features hw-verify, or pass --allow-offline for explicit offline/development checks"
    ))
}

fn next_arg(args: &[String], i: &mut usize, flag: &str) -> anyhow::Result<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn split_csv_arg(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

fn normalized_optional_string_arg(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn apply_image_receipt_prompt_absence_default(
    expected: &mut ExpectedReceipt,
    receipt_request: Option<&ReceiptRequest>,
) {
    let Some(ReceiptRequest::Chat(request)) = receipt_request else {
        return;
    };
    if !request.has_image_inputs()
        || expected.effective_prompt_digest.is_some()
        || expected.effective_prompt_backend.is_some()
        || expected.effective_prompt_kind.is_some()
    {
        return;
    }
    expected.effective_prompt_absent = true;
}

fn parse_u32_arg(flag: &str, value: &str) -> anyhow::Result<u32> {
    value
        .parse::<u32>()
        .map_err(|e| anyhow::anyhow!("{flag} must be an unsigned integer: {e}"))
}

fn parse_u64_arg(flag: &str, value: &str) -> anyhow::Result<u64> {
    value
        .parse::<u64>()
        .map_err(|e| anyhow::anyhow!("{flag} must be an unsigned integer: {e}"))
}

fn parse_tensor_split_arg(value: &str) -> anyhow::Result<Vec<f32>> {
    split_csv_arg(value)
        .into_iter()
        .map(|part| {
            part.parse::<f32>()
                .map_err(|e| anyhow::anyhow!("--tensor-split contains invalid float {part:?}: {e}"))
        })
        .collect()
}

fn parse_receipt_request_type_arg(value: &str) -> anyhow::Result<ReceiptRequestType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "chat-completion" | "chat" => Ok(ReceiptRequestType::ChatCompletion),
        "text-completion" | "completion" | "text" => Ok(ReceiptRequestType::TextCompletion),
        other => Err(anyhow::anyhow!(
            "unknown --receipt-request-type {other:?}; expected chat-completion or text-completion"
        )),
    }
}

fn gpu_execution_digest_from_opts(opts: &CliOpts) -> anyhow::Result<String> {
    let gpu_layers = opts.digest_gpu_layers.ok_or_else(|| {
        anyhow::anyhow!(
            "--print-gpu-execution-digest requires --gpu-layers with the final attested value"
        )
    })?;
    let digest = canonical_gpu_execution_digest(&GpuConfig {
        gpu_layers,
        main_gpu: opts.digest_main_gpu,
        tensor_split: opts.digest_tensor_split.clone(),
    })
    .map_err(|e| anyhow::anyhow!("failed to compute GPU execution digest: {e}"))?;
    Ok(hex::encode(digest))
}

fn decode_optional_hex_arg(name: &str, value: Option<&str>) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let hex = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    if hex.len() != 64 {
        anyhow::bail!(
            "{name} must be a 64-character SHA-256 hex digest, got {} characters",
            hex.len()
        );
    }
    if !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        anyhow::bail!("{name} must contain only hexadecimal characters");
    }
    hex::decode(hex)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("invalid {name} hex: {e}"))
}

fn print_help() {
    println!(
        r#"a3s-power-verify — verify TEE attestation reports from a3s-power

USAGE:
    a3s-power-verify [OPTIONS]

OPTIONS:
    --url <URL>                    Fetch report from a live server (e.g. http://localhost:11434)
    --file <PATH>                  Read report from a JSON file
    --receipt-file <PATH>          Read an AttestationReceipt JSON file and bind it to the report
    --receipt-chat-request-file <PATH> Original ChatCompletionRequest JSON to compare with the receipt
    --receipt-completion-request-file <PATH> Original CompletionRequest JSON to compare with the receipt
                                  Image-bearing chat request files require effective_prompt absence unless explicitly pinned
    --model <NAME>                 Model name to bind into the attestation request
    --nonce <HEX>                  Client nonce to verify (hex-encoded)
    --model-hash <HEX>             Expected model SHA-256 hash (hex-encoded, 32 bytes)
    --expected-measurement <HEX>   Expected 48-byte platform measurement; required unless --allow-offline
    --gpu-evidence-digest <HEX>    Expected SHA-256 digest of NVIDIA GPU CC evidence
    --gpu-verdict-digest <HEX>     Expected SHA-256 digest of NVIDIA NRAS verdict
    --gpu-provider <NAME>          Expected GPU evidence provider label
    --gpu-evidence-format <FORMAT> Expected GPU evidence byte-format label
    --gpu-verdict-format <FORMAT>  Expected GPU verdict byte-format label
    --gpu-evidence-count <N>       Expected number of GPU evidence entries
    --gpu-confidential             Require production NVIDIA GPU CC profile plus evidence nonce, NRAS verdict digest, evidence metadata, topology, claims-version, and identity/version pins
    --require-gpu-evidence         Require a v2 GPU evidence claim and digest verification; requires --nonce
    --require-gpu-device-claims    Require structured NVIDIA GPU/NVSwitch identity and freshness claims; requires --nonce
    --gpu-count <N>                Expected number of NVIDIA GPU device claims
    --nvswitch-count <N>           Expected number of NVIDIA NVSwitch device claims
    --nvswitch-ueid <UEID>         Expected exact NVIDIA NVSwitch UEID set; repeat or comma-separate
    --nvswitch-oemid <ID>          Allowed NVIDIA NVSwitch OEM ID; repeat or comma-separate
    --nvswitch-claims-version <V>  Allowed NVIDIA NVSwitch claims schema version; repeat or comma-separate
    --nvswitch-hwmodel <MODEL>     Allowed NVIDIA NVSwitch hardware model; repeat or comma-separate
    --nvswitch-firmware-version <V> Allowed NVIDIA NVSwitch firmware/BIOS version; repeat or comma-separate
    --gpu-ueid <UEID>              Expected exact NVIDIA GPU UEID set; repeat or comma-separate for multi-GPU
    --gpu-oemid <ID>               Allowed NVIDIA GPU OEM ID; repeat or comma-separate
    --gpu-claims-version <VER>     Allowed NVIDIA GPU claims schema version; repeat or comma-separate
    --gpu-hwmodel <MODEL>          Allowed NVIDIA GPU hardware model; repeat or comma-separate
    --gpu-driver-version <VER>     Allowed NVIDIA GPU driver version; repeat or comma-separate
    --gpu-firmware-version <VER>   Allowed NVIDIA GPU firmware/VBIOS version; repeat or comma-separate
    --chat-template-digest <HEX>   Expected SHA-256 digest of the chat template string
    --decoding-policy-digest <HEX> Expected SHA-256 digest of default decoding parameters
    --gpu-execution-digest <HEX>   Expected SHA-256 digest of GPU execution/offload parameters
    --print-gpu-execution-digest   Print the canonical GPU execution/offload digest and exit
    --gpu-layers <N>               GPU layer offload value for --print-gpu-execution-digest
    --main-gpu <N>                 Main GPU index for --print-gpu-execution-digest (default: 0)
    --tensor-split <CSV>           Tensor split floats for --print-gpu-execution-digest
    --require-runtime-policy       Require a v2 runtime policy claim and digest verification
    --receipt-digest <HEX>         Expected SHA-256 digest of the canonical receipt; requires --receipt-file
    --receipt-model <NAME>         Expected model name in the receipt; requires --receipt-file
    --receipt-request-type <TYPE>  Expected receipt request type: chat-completion or text-completion
    --receipt-input-digest <HEX>   Expected SHA-256 digest of receipt prompt-bearing input
    --receipt-decoding-parameters-digest <HEX> Expected SHA-256 digest of receipt decoding parameters
    --receipt-stream-options-digest <HEX> Expected SHA-256 digest of receipt stream-options JSON
    --receipt-stop-tokens-digest <HEX> Expected SHA-256 digest of receipt stop-token JSON
    --receipt-response-format-digest <HEX> Expected SHA-256 digest of receipt response-format JSON
    --receipt-tools-digest <HEX>   Expected SHA-256 digest of receipt tools JSON
    --receipt-tool-choice-digest <HEX> Expected SHA-256 digest of receipt tool-choice JSON
    --effective-prompt-digest <HEX> Expected SHA-256 digest of the backend effective prompt representation; requires --receipt-file
    --require-effective-prompt-absent Require the receipt to omit effective_prompt
    --effective-prompt-backend <NAME> Expected backend label for the effective prompt digest
    --effective-prompt-kind <KIND> Expected semantic kind for the effective prompt digest
    --allow-offline                Explicitly skip hardware signature verification
    --hw-cert-cache-ttl-secs <N>   AMD KDS / Intel PCS certificate cache TTL in seconds (default: 3600; 0 refetches every verification)
    --help                         Show this help

EXAMPLES:
    # Verify against a live server with nonce and model hash
    a3s-power-verify --url http://localhost:11434 \
        --model llama3 \
        --nonce deadbeef01234567 \
        --model-hash <64-char-hex>

    # Offline/development verification of a saved report file
    a3s-power-verify --file report.json --nonce deadbeef --allow-offline

    # Check measurement only
    a3s-power-verify --file report.json \
        --expected-measurement <96-char-hex>

    # Require NVIDIA GPU confidential-computing evidence binding
    a3s-power-verify --file report.json \
        --nonce deadbeef01234567 \
        --gpu-verdict-digest <64-char-hex> \
        --gpu-provider nvidia-nras \
        --gpu-evidence-format nvidia-nvattest-evidence-json \
        --gpu-verdict-format nvidia-nvattest-attestation-json \
        --gpu-evidence-count <N> \
        --require-gpu-evidence

    # Also require structured NVIDIA device identity/freshness claims
    a3s-power-verify --file report.json \
        --nonce deadbeef01234567 \
        --gpu-verdict-digest <64-char-hex> \
        --gpu-provider nvidia-nras \
        --gpu-evidence-format nvidia-nvattest-evidence-json \
        --gpu-verdict-format nvidia-nvattest-attestation-json \
        --gpu-evidence-count <N> \
        --require-gpu-device-claims \
        --gpu-count 1 \
        --nvswitch-count 0 \
        --gpu-ueid 655333107904478077882826344426270545524203067314 \
        --gpu-oemid 5703 \
        --gpu-claims-version 3.0 \
        --gpu-hwmodel "GH100 A01 GSP BROM" \
        --gpu-driver-version 590.12 \
        --gpu-firmware-version 96.00.A5.00.01

    # Production NVIDIA GPU confidential-computing profile
    a3s-power-verify --file report.json \
        --expected-measurement <96-char-hex> \
        --nonce <64-char-hex> \
        --gpu-confidential \
        --gpu-verdict-digest <64-char-hex> \
        --gpu-provider nvidia-nras \
        --gpu-evidence-format nvidia-nvattest-evidence-json \
        --gpu-verdict-format nvidia-nvattest-attestation-json \
        --gpu-evidence-count <N> \
        --gpu-execution-digest <64-char-hex> \
        --gpu-count 1 \
        --nvswitch-count 0 \
        --gpu-ueid 655333107904478077882826344426270545524203067314 \
        --gpu-oemid 5703 \
        --gpu-claims-version 3.0 \
        --gpu-hwmodel "GH100 A01 GSP BROM" \
        --gpu-driver-version 590.12 \
        --gpu-firmware-version 96.00.A5.00.01

    # Require runtime prompt/template policy binding
    a3s-power-verify --file report.json \
        --chat-template-digest <64-char-hex> \
        --require-runtime-policy

    # Compute the GPU execution/offload digest to use with --gpu-execution-digest
    a3s-power-verify --print-gpu-execution-digest \
        --gpu-layers -1 \
        --main-gpu 0 \
        --tensor-split 0.5,0.5

    # Verify an inference receipt against the saved attestation report
    a3s-power-verify --file report.json \
        --receipt-file receipt.json \
        --receipt-chat-request-file chat-request.json \
        --receipt-digest <64-char-hex> \
        --receipt-model llama3 \
        --receipt-request-type chat-completion \
        --receipt-input-digest <64-char-hex> \
        --receipt-decoding-parameters-digest <64-char-hex> \
        --receipt-stream-options-digest <64-char-hex> \
        --receipt-stop-tokens-digest <64-char-hex> \
        --allow-offline
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

fn load_receipt(opts: &CliOpts) -> anyhow::Result<Option<AttestationReceipt>> {
    let Some(path) = opts.receipt_file.as_deref() else {
        return Ok(None);
    };

    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read receipt file {path}: {e}"))?;
    let receipt: AttestationReceipt = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("failed to parse receipt JSON: {e}"))?;
    Ok(Some(receipt))
}

fn load_receipt_request(opts: &CliOpts) -> anyhow::Result<Option<ReceiptRequest>> {
    if let Some(path) = opts.receipt_chat_request_file.as_deref() {
        let request = load_json_file::<ChatCompletionRequest>(path, "chat request")?;
        return Ok(Some(ReceiptRequest::Chat(request)));
    }

    if let Some(path) = opts.receipt_completion_request_file.as_deref() {
        let request = load_json_file::<CompletionRequest>(path, "completion request")?;
        return Ok(Some(ReceiptRequest::Completion(request)));
    }

    Ok(None)
}

fn load_json_file<T>(path: &str, label: &str) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {label} file {path}: {e}"))?;
    serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("failed to parse {label} JSON: {e}"))
}

fn fetch_report(
    base_url: &str,
    model: Option<&str>,
    nonce: Option<&str>,
) -> anyhow::Result<AttestationReport> {
    let url = attestation_url(base_url, model, nonce)?;

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

fn attestation_url(
    base_url: &str,
    model: Option<&str>,
    nonce: Option<&str>,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(&format!("{}/", base_url.trim_end_matches('/')))
        .map_err(|e| anyhow::anyhow!("invalid --url {base_url:?}: {e}"))?;
    url.set_query(None);
    url.set_fragment(None);
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("--url {base_url:?} cannot be used as a base URL"))?;
        path.pop_if_empty();
        path.push("v1");
        path.push("attestation");
    }
    {
        let mut query = url.query_pairs_mut();
        if let Some(nonce) = nonce {
            query.append_pair("nonce", nonce);
        }
        if let Some(model) = model {
            query.append_pair("model", model);
        }
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use a3s_power::api::receipt::{
        chat_receipt_with_runtime_policy, chat_receipt_with_runtime_policy_and_effective_prompt,
        completion_receipt_with_runtime_policy, receipt_decoding_parameters_digest, receipt_digest,
        ReceiptDecodingPolicy, ReceiptInputDigest, ReceiptRequestType,
    };
    use a3s_power::api::types::{ChatCompletionMessage, ChatCompletionRequest, CompletionRequest};
    use a3s_power::backend::types::{EffectivePromptDigest, MessageContent};
    use a3s_power::tee::attestation::TeeType;
    use a3s_power::tee::attestation::{
        build_claims_report_data, AttestationClaimsV2, DecodingPolicyClaim, RuntimePolicyClaim,
    };
    use std::collections::BTreeMap;

    fn write_report_file() -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        let report = AttestationReport {
            version: "1.0".to_string(),
            tee_type: TeeType::Simulated,
            report_data: vec![0u8; 64],
            measurement: vec![0u8; 48],
            raw_report: None,
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: None,
        };
        std::fs::write(file.path(), serde_json::to_vec(&report).unwrap()).unwrap();
        file
    }

    fn runtime_policy() -> RuntimePolicyClaim {
        RuntimePolicyClaim::new().with_decoding(DecodingPolicyClaim {
            parameters_sha256: vec![0x44; 32],
        })
    }

    fn write_report_and_receipt_files() -> (tempfile::NamedTempFile, tempfile::NamedTempFile, String)
    {
        let report_file = tempfile::NamedTempFile::new().unwrap();
        let receipt_file = tempfile::NamedTempFile::new().unwrap();
        let runtime_policy = runtime_policy();
        let claims =
            AttestationClaimsV2::new(TeeType::Simulated).with_runtime(runtime_policy.clone());
        let report = AttestationReport {
            version: "1.0".to_string(),
            tee_type: TeeType::Simulated,
            report_data: build_claims_report_data(&claims).unwrap(),
            measurement: vec![0u8; 48],
            raw_report: None,
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: Some(claims),
        };
        let receipt = AttestationReceipt {
            schema: AttestationReceipt::SCHEMA.to_string(),
            request_type: ReceiptRequestType::ChatCompletion,
            model: "test-model".to_string(),
            input: ReceiptInputDigest {
                kind: "chat.messages".to_string(),
                sha256: "00".repeat(32),
            },
            runtime_policy: Some(runtime_policy),
            effective_prompt: None,
            decoding: ReceiptDecodingPolicy {
                parameters: BTreeMap::from([("temperature".to_string(), serde_json::json!(0.2))]),
                stream_options_sha256: Some("55".repeat(32)),
                stop_tokens_sha256: Some("11".repeat(32)),
                response_format_sha256: Some("22".repeat(32)),
                tools_sha256: Some("33".repeat(32)),
                tool_choice_sha256: Some("44".repeat(32)),
            },
        };
        let digest = receipt_digest(&receipt).unwrap();

        std::fs::write(report_file.path(), serde_json::to_vec(&report).unwrap()).unwrap();
        std::fs::write(receipt_file.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();

        (report_file, receipt_file, digest)
    }

    fn chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
                thinking: None,
            }],
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(128),
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: Some(vec!["</s>".to_string()]),
            stream: Some(false),
            stream_options: None,
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.0),
            seed: Some(7),
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            keep_alive: None,
        }
    }

    fn image_chat_request() -> ChatCompletionRequest {
        let mut request = chat_request();
        request.messages[0].images = Some(vec!["aGVsbG8=".to_string()]);
        request
    }

    fn completion_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(128),
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: Some(vec!["</s>".to_string()]),
            stream: Some(false),
            stream_options: None,
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.0),
            seed: Some(7),
            keep_alive: None,
        }
    }

    fn write_runtime_report_file(runtime_policy: RuntimePolicyClaim) -> tempfile::NamedTempFile {
        let report_file = tempfile::NamedTempFile::new().unwrap();
        let claims = AttestationClaimsV2::new(TeeType::Simulated).with_runtime(runtime_policy);
        let report = AttestationReport {
            version: "1.0".to_string(),
            tee_type: TeeType::Simulated,
            report_data: build_claims_report_data(&claims).unwrap(),
            measurement: vec![0u8; 48],
            raw_report: None,
            timestamp: chrono::Utc::now(),
            nonce: None,
            claims: Some(claims),
        };
        std::fs::write(report_file.path(), serde_json::to_vec(&report).unwrap()).unwrap();
        report_file
    }

    fn write_report_receipt_and_chat_request_files() -> (
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
    ) {
        let receipt_file = tempfile::NamedTempFile::new().unwrap();
        let request_file = tempfile::NamedTempFile::new().unwrap();
        let runtime_policy = runtime_policy();
        let report_file = write_runtime_report_file(runtime_policy.clone());
        let request = chat_request();
        let receipt = chat_receipt_with_runtime_policy(&request, Some(runtime_policy)).unwrap();

        std::fs::write(receipt_file.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();
        std::fs::write(request_file.path(), serde_json::to_vec(&request).unwrap()).unwrap();

        (report_file, receipt_file, request_file)
    }

    fn write_report_receipt_and_image_chat_request_files() -> (
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
        EffectivePromptDigest,
    ) {
        let receipt_file = tempfile::NamedTempFile::new().unwrap();
        let request_file = tempfile::NamedTempFile::new().unwrap();
        let runtime_policy = runtime_policy();
        let report_file = write_runtime_report_file(runtime_policy.clone());
        let request = image_chat_request();
        let effective_prompt =
            EffectivePromptDigest::chat_rendered_prompt("test-backend", "rendered image prompt");
        let receipt = chat_receipt_with_runtime_policy_and_effective_prompt(
            &request,
            Some(runtime_policy),
            Some(effective_prompt.clone()),
        )
        .unwrap();

        std::fs::write(receipt_file.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();
        std::fs::write(request_file.path(), serde_json::to_vec(&request).unwrap()).unwrap();

        (report_file, receipt_file, request_file, effective_prompt)
    }

    fn write_report_receipt_and_image_chat_request_without_effective_prompt_files() -> (
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
    ) {
        let receipt_file = tempfile::NamedTempFile::new().unwrap();
        let request_file = tempfile::NamedTempFile::new().unwrap();
        let runtime_policy = runtime_policy();
        let report_file = write_runtime_report_file(runtime_policy.clone());
        let request = image_chat_request();
        let receipt = chat_receipt_with_runtime_policy(&request, Some(runtime_policy)).unwrap();
        assert!(receipt.effective_prompt.is_none());

        std::fs::write(receipt_file.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();
        std::fs::write(request_file.path(), serde_json::to_vec(&request).unwrap()).unwrap();

        (report_file, receipt_file, request_file)
    }

    fn write_report_receipt_and_completion_request_files() -> (
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
        tempfile::NamedTempFile,
    ) {
        let receipt_file = tempfile::NamedTempFile::new().unwrap();
        let request_file = tempfile::NamedTempFile::new().unwrap();
        let runtime_policy = runtime_policy();
        let report_file = write_runtime_report_file(runtime_policy.clone());
        let request = completion_request();
        let receipt =
            completion_receipt_with_runtime_policy(&request, Some(runtime_policy)).unwrap();

        std::fs::write(receipt_file.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();
        std::fs::write(request_file.path(), serde_json::to_vec(&request).unwrap()).unwrap();

        (report_file, receipt_file, request_file)
    }

    #[test]
    fn test_attestation_url_encodes_query_params() {
        let url = attestation_url(
            "http://localhost:11434/",
            Some("A3S-Lab/model & version=1"),
            Some("001122"),
        )
        .unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let query: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();

        assert_eq!(
            parsed.as_str().split('?').next().unwrap(),
            "http://localhost:11434/v1/attestation"
        );
        assert_eq!(
            query,
            vec![
                ("nonce".to_string(), "001122".to_string()),
                ("model".to_string(), "A3S-Lab/model & version=1".to_string()),
            ]
        );
        assert!(url.contains("model=A3S-Lab%2Fmodel+%26+version%3D1"));
    }

    #[test]
    fn test_attestation_url_preserves_base_path_and_drops_query_fragment() {
        let url = attestation_url(
            "http://localhost:11434/api?stale=1#fragment",
            Some("llama3"),
            Some("001122"),
        )
        .unwrap();

        assert_eq!(
            url,
            "http://localhost:11434/api/v1/attestation?nonce=001122&model=llama3"
        );
    }

    #[test]
    fn test_parse_args_defaults_to_strict_online_verification() {
        let args = vec!["--file".to_string(), "report.json".to_string()];
        let opts = parse_args(&args).unwrap();
        assert_eq!(opts.file.as_deref(), Some("report.json"));
        assert!(!opts.allow_offline);
    }

    #[test]
    fn test_parse_args_allow_offline_is_explicit() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--allow-offline".to_string(),
        ];
        let opts = parse_args(&args).unwrap();
        assert!(opts.allow_offline);
    }

    #[test]
    fn test_run_default_requires_expected_measurement() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--expected-measurement"));
    }

    #[test]
    fn test_parse_args_gpu_evidence_policy() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--require-gpu-evidence".to_string(),
        ];
        let opts = parse_args(&args).unwrap();
        assert_eq!(
            opts.gpu_verdict_digest.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert!(opts.require_gpu_evidence);
    }

    #[test]
    fn test_parse_args_gpu_confidential_profile() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-confidential".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert!(opts.gpu_confidential);
    }

    #[test]
    fn test_parse_args_gpu_evidence_metadata_pins() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "8".to_string(),
        ];
        let opts = parse_args(&args).unwrap();

        assert!(opts.has_gpu_evidence_pins());
        assert!(opts.has_gpu_confidential_evidence_pins());
        let expected = opts.expected_gpu_evidence().unwrap();
        assert_eq!(expected.provider.as_deref(), Some("nvidia-nras"));
        assert_eq!(
            expected.evidence_format.as_deref(),
            Some("nvidia-nvattest-evidence-json")
        );
        assert_eq!(
            expected.verdict_format.as_deref(),
            Some("nvidia-nvattest-attestation-json")
        );
        assert_eq!(expected.evidence_count, Some(8));
    }

    #[test]
    fn test_parse_args_gpu_evidence_count_rejects_invalid_value() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-evidence-count".to_string(),
            "many".to_string(),
        ];

        let err = parse_args(&args).unwrap_err();

        assert!(err.to_string().contains("must be an unsigned integer"));
    }

    #[test]
    fn test_parse_args_gpu_evidence_count_rejects_zero() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-evidence-count".to_string(),
            "0".to_string(),
        ];

        let err = parse_args(&args).unwrap_err();

        assert!(err.to_string().contains("must be greater than zero"));
    }

    #[test]
    fn test_parse_args_gpu_device_claims_policy() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--require-gpu-device-claims".to_string(),
        ];
        let opts = parse_args(&args).unwrap();

        assert!(opts.require_gpu_device_claims);
    }

    #[test]
    fn test_parse_args_gpu_identity_pins() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-count".to_string(),
            "2".to_string(),
            "--nvswitch-count".to_string(),
            "1".to_string(),
            "--nvswitch-ueid".to_string(),
            "nvswitch-1".to_string(),
            "--nvswitch-oemid".to_string(),
            "5703".to_string(),
            "--nvswitch-claims-version".to_string(),
            "3.0".to_string(),
            "--nvswitch-hwmodel".to_string(),
            "NVSwitch B01".to_string(),
            "--nvswitch-firmware-version".to_string(),
            "1.2.3".to_string(),
            "--gpu-ueid".to_string(),
            "ueid-1,ueid-2".to_string(),
            "--gpu-oemid".to_string(),
            "5703".to_string(),
            "--gpu-claims-version".to_string(),
            "3.0".to_string(),
            "--gpu-hwmodel".to_string(),
            "GH100 A01 GSP BROM".to_string(),
            "--gpu-driver-version".to_string(),
            "590.12".to_string(),
            "--gpu-firmware-version".to_string(),
            "96.00.A5.00.01".to_string(),
        ];
        let opts = parse_args(&args).unwrap();

        assert!(opts.has_gpu_device_pins());
        assert!(opts.has_gpu_confidential_device_pins());
        assert_eq!(opts.gpu_count, Some(2));
        assert_eq!(opts.nvswitch_count, Some(1));
        assert_eq!(opts.nvswitch_ueids, vec!["nvswitch-1"]);
        assert_eq!(opts.nvswitch_oemids, vec!["5703"]);
        assert_eq!(opts.nvswitch_claims_versions, vec!["3.0"]);
        assert_eq!(opts.nvswitch_hwmodels, vec!["NVSwitch B01"]);
        assert_eq!(opts.nvswitch_firmware_versions, vec!["1.2.3"]);
        assert_eq!(opts.gpu_ueids, vec!["ueid-1", "ueid-2"]);
        assert_eq!(opts.gpu_oemids, vec!["5703"]);
        assert_eq!(opts.gpu_claims_versions, vec!["3.0"]);
        assert_eq!(opts.gpu_hwmodels, vec!["GH100 A01 GSP BROM"]);
        assert_eq!(opts.gpu_driver_versions, vec!["590.12"]);
        assert_eq!(opts.gpu_firmware_versions, vec!["96.00.A5.00.01"]);
        let expected = opts.expected_gpu_devices().unwrap();
        assert_eq!(expected.gpu_count, Some(2));
        assert_eq!(expected.nvswitch_count, Some(1));
        assert_eq!(expected.nvswitch_ueids, vec!["nvswitch-1"]);
        assert_eq!(expected.nvswitch_oemids, vec!["5703"]);
        assert_eq!(expected.nvswitch_claims_versions, vec!["3.0"]);
        assert_eq!(expected.nvswitch_hwmodels, vec!["NVSwitch B01"]);
        assert_eq!(expected.nvswitch_firmware_versions, vec!["1.2.3"]);
        assert_eq!(expected.oemids, vec!["5703"]);
        assert_eq!(expected.claims_versions, vec!["3.0"]);
    }

    #[test]
    fn test_parse_args_gpu_count_rejects_zero() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--gpu-count".to_string(),
            "0".to_string(),
        ];

        let err = parse_args(&args).unwrap_err();

        assert!(err.to_string().contains("must be greater than zero"));
    }

    #[test]
    fn test_parse_args_nvswitch_count_accepts_zero() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--nvswitch-count".to_string(),
            "0".to_string(),
        ];
        let opts = parse_args(&args).unwrap();

        assert_eq!(opts.nvswitch_count, Some(0));
    }

    #[test]
    fn test_run_require_gpu_evidence_requires_nonce() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--require-gpu-evidence".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();
        assert!(err.to_string().contains("require --nonce"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_nonce() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("require --nonce"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_32_byte_nonce() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "01020304".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("64 hex characters"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_evidence_metadata_pins() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-hwmodel".to_string(),
            "GH100 A01 GSP BROM".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-provider"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_verdict_digest() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-evidence-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
            "--gpu-claims-version".to_string(),
            "3.0".to_string(),
            "--gpu-count".to_string(),
            "1".to_string(),
            "--gpu-hwmodel".to_string(),
            "GH100 A01 GSP BROM".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-verdict-digest"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_gpu_execution_digest() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-execution-digest"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_identity_pin() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-count"));
    }

    #[test]
    fn test_run_gpu_confidential_treats_oemid_as_supplemental_pin() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-claims-version".to_string(),
            "3.0".to_string(),
            "--gpu-oemid".to_string(),
            "5703".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-count"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_nvswitch_identity_pins_when_present() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-claims-version".to_string(),
            "3.0".to_string(),
            "--gpu-ueid".to_string(),
            "gpu-ueid-0".to_string(),
            "--nvswitch-count".to_string(),
            "1".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--nvswitch-claims-version"));
    }

    #[test]
    fn test_run_gpu_confidential_requires_claims_version_pin() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--nonce".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--gpu-confidential".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--gpu-evidence-format".to_string(),
            "nvidia-nvattest-evidence-json".to_string(),
            "--gpu-verdict-format".to_string(),
            "nvidia-nvattest-attestation-json".to_string(),
            "--gpu-evidence-count".to_string(),
            "1".to_string(),
            "--gpu-execution-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-count".to_string(),
            "1".to_string(),
            "--gpu-hwmodel".to_string(),
            "GH100 A01 GSP BROM".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("--gpu-claims-version"));
    }

    #[test]
    fn test_run_gpu_evidence_metadata_pins_require_nonce() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--gpu-provider".to_string(),
            "nvidia-nras".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();
        assert!(err
            .to_string()
            .contains("GPU evidence/device pins require --nonce"));
    }

    #[test]
    fn test_run_gpu_identity_pins_require_nonce() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--gpu-ueid".to_string(),
            "ueid-1".to_string(),
            "--gpu-verdict-digest".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();
        assert!(err
            .to_string()
            .contains("GPU evidence/device pins require --nonce"));
    }

    #[test]
    fn test_run_gpu_verdict_digest_rejects_short_hex() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--gpu-verdict-digest".to_string(),
            "abcd".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--gpu-verdict-digest must be a 64-character SHA-256 hex digest"));
    }

    #[test]
    fn test_run_gpu_execution_digest_rejects_non_hex() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--gpu-execution-digest".to_string(),
            "g".repeat(64),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--gpu-execution-digest must contain only hexadecimal characters"));
    }

    #[test]
    fn test_parse_args_runtime_policy() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--chat-template-digest".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "--gpu-execution-digest".to_string(),
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "--require-runtime-policy".to_string(),
        ];
        let opts = parse_args(&args).unwrap();
        assert_eq!(
            opts.chat_template_digest.as_deref(),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        );
        assert_eq!(
            opts.gpu_execution_digest.as_deref(),
            Some("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        );
        assert!(opts.require_runtime_policy);
    }

    #[test]
    fn test_parse_args_print_gpu_execution_digest() {
        let args = vec![
            "--print-gpu-execution-digest".to_string(),
            "--gpu-layers".to_string(),
            "-1".to_string(),
            "--main-gpu".to_string(),
            "1".to_string(),
            "--tensor-split".to_string(),
            "0.25,0.75".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert!(opts.print_gpu_execution_digest);
        assert_eq!(opts.digest_gpu_layers, Some(-1));
        assert_eq!(opts.digest_main_gpu, 1);
        assert_eq!(opts.digest_tensor_split, vec![0.25, 0.75]);
    }

    #[test]
    fn test_gpu_execution_digest_from_opts_matches_canonicalizer() {
        let args = vec![
            "--print-gpu-execution-digest".to_string(),
            "--gpu-layers".to_string(),
            "-1".to_string(),
            "--main-gpu".to_string(),
            "0".to_string(),
            "--tensor-split".to_string(),
            "0.5,0.5".to_string(),
        ];
        let opts = parse_args(&args).unwrap();
        let expected = hex::encode(
            canonical_gpu_execution_digest(&GpuConfig {
                gpu_layers: -1,
                main_gpu: 0,
                tensor_split: vec![0.5, 0.5],
            })
            .unwrap(),
        );

        assert_eq!(gpu_execution_digest_from_opts(&opts).unwrap(), expected);
    }

    #[test]
    fn test_gpu_execution_digest_from_opts_requires_gpu_layers() {
        let args = vec!["--print-gpu-execution-digest".to_string()];
        let opts = parse_args(&args).unwrap();

        let err = gpu_execution_digest_from_opts(&opts).unwrap_err();

        assert!(err.to_string().contains("--gpu-layers"));
    }

    #[test]
    fn test_parse_args_receipt_file_and_digest() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--receipt-file".to_string(),
            "receipt.json".to_string(),
            "--receipt-digest".to_string(),
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "--effective-prompt-digest".to_string(),
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert_eq!(opts.receipt_file.as_deref(), Some("receipt.json"));
        assert_eq!(
            opts.receipt_digest.as_deref(),
            Some("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        );
        assert_eq!(
            opts.effective_prompt_digest.as_deref(),
            Some("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        );
    }

    #[test]
    fn test_parse_args_receipt_request_files() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--receipt-file".to_string(),
            "receipt.json".to_string(),
            "--receipt-chat-request-file".to_string(),
            "chat-request.json".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert_eq!(
            opts.receipt_chat_request_file.as_deref(),
            Some("chat-request.json")
        );
        assert!(opts.receipt_completion_request_file.is_none());
        assert!(opts.has_receipt_request_file());
    }

    #[test]
    fn test_parse_args_hw_cert_cache_ttl_secs() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--hw-cert-cache-ttl-secs".to_string(),
            "0".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert_eq!(opts.hw_cert_cache_ttl_secs, Some(0));
    }

    #[test]
    fn test_parse_args_rejects_invalid_hw_cert_cache_ttl_secs() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--hw-cert-cache-ttl-secs".to_string(),
            "not-a-number".to_string(),
        ];

        let err = parse_args(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--hw-cert-cache-ttl-secs must be an unsigned integer"));
    }

    #[test]
    fn test_parse_args_receipt_policy_pins() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--receipt-file".to_string(),
            "receipt.json".to_string(),
            "--receipt-model".to_string(),
            "llama3".to_string(),
            "--receipt-request-type".to_string(),
            "chat-completion".to_string(),
            "--receipt-input-digest".to_string(),
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string(),
            "--receipt-decoding-parameters-digest".to_string(),
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string(),
            "--receipt-stream-options-digest".to_string(),
            "5555555555555555555555555555555555555555555555555555555555555555".to_string(),
            "--receipt-stop-tokens-digest".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--receipt-response-format-digest".to_string(),
            "2222222222222222222222222222222222222222222222222222222222222222".to_string(),
            "--receipt-tools-digest".to_string(),
            "3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            "--receipt-tool-choice-digest".to_string(),
            "4444444444444444444444444444444444444444444444444444444444444444".to_string(),
            "--effective-prompt-digest".to_string(),
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "--effective-prompt-backend".to_string(),
            "llama.cpp".to_string(),
            "--effective-prompt-kind".to_string(),
            "chat.rendered-prompt".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert!(opts.has_receipt_policy_pins());
        assert_eq!(opts.receipt_model.as_deref(), Some("llama3"));
        assert_eq!(
            opts.receipt_request_type,
            Some(ReceiptRequestType::ChatCompletion)
        );
        assert_eq!(
            opts.receipt_input_digest.as_deref(),
            Some("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        );
        assert_eq!(
            opts.receipt_decoding_parameters_digest.as_deref(),
            Some("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
        );
        assert_eq!(
            opts.receipt_stream_options_digest.as_deref(),
            Some("5555555555555555555555555555555555555555555555555555555555555555")
        );
        assert_eq!(
            opts.receipt_stop_tokens_digest.as_deref(),
            Some("1111111111111111111111111111111111111111111111111111111111111111")
        );
        assert_eq!(
            opts.receipt_response_format_digest.as_deref(),
            Some("2222222222222222222222222222222222222222222222222222222222222222")
        );
        assert_eq!(
            opts.receipt_tools_digest.as_deref(),
            Some("3333333333333333333333333333333333333333333333333333333333333333")
        );
        assert_eq!(
            opts.receipt_tool_choice_digest.as_deref(),
            Some("4444444444444444444444444444444444444444444444444444444444444444")
        );
        assert_eq!(
            opts.effective_prompt_digest.as_deref(),
            Some("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        );
        assert_eq!(opts.effective_prompt_backend.as_deref(), Some("llama.cpp"));
        assert_eq!(
            opts.effective_prompt_kind.as_deref(),
            Some("chat.rendered-prompt")
        );

        let expected = opts.expected_receipt(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![0xdd; 32]),
        );
        assert_eq!(expected.effective_prompt_digest, Some(vec![0xdd; 32]));
    }

    #[test]
    fn test_parse_args_require_effective_prompt_absent() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--receipt-file".to_string(),
            "receipt.json".to_string(),
            "--require-effective-prompt-absent".to_string(),
        ];

        let opts = parse_args(&args).unwrap();

        assert!(opts.effective_prompt_absent);
        assert!(opts.has_receipt_policy_pins());
        let expected = opts.expected_receipt(None, None, None, None, None, None, None, None);
        assert!(expected.effective_prompt_absent);
    }

    #[test]
    fn test_expected_receipt_trims_effective_prompt_backend_and_kind() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--receipt-file".to_string(),
            "receipt.json".to_string(),
            "--effective-prompt-backend".to_string(),
            " llama.cpp ".to_string(),
            "--effective-prompt-kind".to_string(),
            " chat.rendered-prompt ".to_string(),
        ];

        let opts = parse_args(&args).unwrap();
        let expected = opts.expected_receipt(None, None, None, None, None, None, None, None);

        assert_eq!(
            expected.effective_prompt_backend.as_deref(),
            Some("llama.cpp")
        );
        assert_eq!(
            expected.effective_prompt_kind.as_deref(),
            Some("chat.rendered-prompt")
        );
    }

    #[test]
    fn test_expected_receipt_ignores_blank_effective_prompt_backend_and_kind() {
        let args = vec![
            "--file".to_string(),
            "report.json".to_string(),
            "--effective-prompt-backend".to_string(),
            "   ".to_string(),
            "--effective-prompt-kind".to_string(),
            "\t".to_string(),
        ];

        let opts = parse_args(&args).unwrap();
        let expected = opts.expected_receipt(None, None, None, None, None, None, None, None);

        assert!(!opts.has_receipt_policy_pins());
        assert!(expected.is_empty());
    }

    #[test]
    fn test_run_receipt_digest_requires_receipt_file() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("require --receipt-file"));
    }

    #[test]
    fn test_run_receipt_digest_rejects_short_hex() {
        let (report_file, receipt_file, _) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            "abcd".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--receipt-digest must be a 64-character SHA-256 hex digest"));
    }

    #[test]
    fn test_run_effective_prompt_digest_rejects_non_hex() {
        let (report_file, receipt_file, _) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--effective-prompt-digest".to_string(),
            "g".repeat(64),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--effective-prompt-digest must contain only hexadecimal characters"));
    }

    #[test]
    fn test_run_model_hash_rejects_short_hex() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--model-hash".to_string(),
            "bbbb".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("--model-hash must be a 64-character SHA-256 hex digest"));
    }

    #[test]
    fn test_run_effective_prompt_absent_requires_receipt_file() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--require-effective-prompt-absent".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("require --receipt-file"));
    }

    #[test]
    fn test_run_rejects_conflicting_effective_prompt_absent_policy() {
        let (report_file, receipt_file, _) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--require-effective-prompt-absent".to_string(),
            "--effective-prompt-digest".to_string(),
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn test_run_receipt_policy_pins_require_receipt_file() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-model".to_string(),
            "test-model".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("require --receipt-file"));
    }

    #[test]
    fn test_run_receipt_request_file_requires_receipt_file() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            "chat-request.json".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("require --receipt-file"));
    }

    #[test]
    fn test_run_rejects_conflicting_receipt_request_files() {
        let (report_file, receipt_file, _) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            "chat-request.json".to_string(),
            "--receipt-completion-request-file".to_string(),
            "completion-request.json".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn test_run_verifies_receipt_against_attestation() {
        let (report_file, receipt_file, receipt_digest) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            receipt_digest,
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_verifies_receipt_chat_request_file() {
        let (report_file, receipt_file, request_file) =
            write_report_receipt_and_chat_request_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            request_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_rejects_image_receipt_effective_prompt_without_explicit_pin() {
        let (report_file, receipt_file, request_file, _) =
            write_report_receipt_and_image_chat_request_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            request_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("effective prompt digest is present"));
    }

    #[test]
    fn test_run_verifies_image_receipt_with_effective_prompt_absent_by_default() {
        let (report_file, receipt_file, request_file) =
            write_report_receipt_and_image_chat_request_without_effective_prompt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            request_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_allows_image_receipt_effective_prompt_with_explicit_pin() {
        let (report_file, receipt_file, request_file, effective_prompt) =
            write_report_receipt_and_image_chat_request_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            request_file.path().display().to_string(),
            "--effective-prompt-digest".to_string(),
            effective_prompt.sha256,
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_verifies_receipt_completion_request_file() {
        let (report_file, receipt_file, request_file) =
            write_report_receipt_and_completion_request_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-completion-request-file".to_string(),
            request_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_verifies_receipt_policy_pins() {
        let (report_file, receipt_file, receipt_digest) = write_report_and_receipt_files();
        let receipt: AttestationReceipt =
            serde_json::from_slice(&std::fs::read(receipt_file.path()).unwrap()).unwrap();
        let decoding_parameters_digest = receipt_decoding_parameters_digest(&receipt).unwrap();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            receipt_digest,
            "--receipt-model".to_string(),
            "test-model".to_string(),
            "--receipt-request-type".to_string(),
            "chat-completion".to_string(),
            "--receipt-input-digest".to_string(),
            "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "--receipt-decoding-parameters-digest".to_string(),
            decoding_parameters_digest,
            "--receipt-stream-options-digest".to_string(),
            "5555555555555555555555555555555555555555555555555555555555555555".to_string(),
            "--receipt-stop-tokens-digest".to_string(),
            "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            "--receipt-response-format-digest".to_string(),
            "2222222222222222222222222222222222222222222222222222222222222222".to_string(),
            "--receipt-tools-digest".to_string(),
            "3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            "--receipt-tool-choice-digest".to_string(),
            "4444444444444444444444444444444444444444444444444444444444444444".to_string(),
            "--require-effective-prompt-absent".to_string(),
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[test]
    fn test_run_rejects_receipt_digest_mismatch() {
        let (report_file, receipt_file, _) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("receipt verification failed"));
    }

    #[test]
    fn test_run_rejects_receipt_policy_input_mismatch() {
        let (report_file, receipt_file, receipt_digest) = write_report_and_receipt_files();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-digest".to_string(),
            receipt_digest,
            "--receipt-input-digest".to_string(),
            "9999999999999999999999999999999999999999999999999999999999999999".to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err.to_string().contains("receipt verification failed"));
        assert!(err.to_string().contains("receipt.input.sha256 mismatch"));
    }

    #[test]
    fn test_run_rejects_receipt_chat_request_mismatch() {
        let (report_file, receipt_file, request_file) =
            write_report_receipt_and_chat_request_files();
        let mut request = chat_request();
        request.messages[0].content = MessageContent::Text("different".to_string());
        std::fs::write(request_file.path(), serde_json::to_vec(&request).unwrap()).unwrap();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--receipt-file".to_string(),
            receipt_file.path().display().to_string(),
            "--receipt-chat-request-file".to_string(),
            request_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        let err = run(&args).unwrap_err();

        assert!(err
            .to_string()
            .contains("receipt request verification failed"));
        assert!(err.to_string().contains("receipt.input.sha256 mismatch"));
    }

    #[test]
    fn test_run_allow_offline_saved_report_passes_without_hardware_verifier() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--allow-offline".to_string(),
        ];

        run(&args).unwrap();
    }

    #[cfg(not(feature = "hw-verify"))]
    #[test]
    fn test_run_default_requires_hw_verify_feature_or_allow_offline() {
        let report_file = write_report_file();
        let args = vec![
            "--file".to_string(),
            report_file.path().display().to_string(),
            "--expected-measurement".to_string(),
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".to_string(),
        ];

        let err = run(&args).unwrap_err();
        assert!(err.to_string().contains("requires building"));
    }
}
