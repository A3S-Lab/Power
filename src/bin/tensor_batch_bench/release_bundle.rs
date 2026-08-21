use std::path::Path;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::ReleaseEvidenceBundle;
use a3s_power::tee::attestation::TeeType;
use serde::Serialize;

use super::{read_bounded_regular, Arguments, MAX_INPUT_DOCUMENT_BYTES};

const MAX_PIN_BYTES: u64 = 66;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReleaseBundleVerification<'a> {
    schema: &'static str,
    verified: bool,
    bundle_sha256: &'a str,
    power_version: &'a str,
    power_commit: &'a str,
    platform_count: usize,
    confidential_tee: TeeType,
}

pub(super) fn run(arguments: &mut Arguments) -> Result<serde_json::Value> {
    let bundle_path = arguments.required_path("--bundle")?;
    let pin_path = arguments.required_path("--expected-sha256-file")?;
    let expected_power_version = arguments.required("--power-version")?;
    let expected_power_commit = arguments.required("--power-commit")?;

    let source = read_bounded_regular(
        &bundle_path,
        MAX_INPUT_DOCUMENT_BYTES,
        "release evidence bundle",
    )?;
    let bundle = serde_json::from_slice::<ReleaseEvidenceBundle>(&source)?;
    let expected_sha256 = read_sha256_pin(&pin_path)?;
    bundle.verify_strict_v1_release(
        &expected_sha256,
        &expected_power_version,
        &expected_power_commit,
    )?;

    serde_json::to_value(ReleaseBundleVerification {
        schema: "a3s.power.release-evidence-verification.v1",
        verified: true,
        bundle_sha256: &bundle.sha256,
        power_version: &bundle.policy.revision.power_version,
        power_commit: &bundle.policy.revision.power_commit,
        platform_count: bundle.captures.len(),
        confidential_tee: TeeType::SevSnp,
    })
    .map_err(PowerError::from)
}

fn read_sha256_pin(path: &Path) -> Result<String> {
    let bytes = read_bounded_regular(path, MAX_PIN_BYTES, "release evidence SHA-256 pin")?;
    let digest = match bytes.as_slice() {
        [digest @ .., b'\r', b'\n'] if digest.len() == 64 => digest,
        [digest @ .., b'\n'] if digest.len() == 64 => digest,
        digest if digest.len() == 64 => digest,
        _ => {
            return Err(PowerError::InvalidFormat(
                "release evidence SHA-256 pin must contain exactly one lowercase digest with an optional line ending"
                    .to_string(),
            ))
        }
    };
    String::from_utf8(digest.to_vec()).map_err(|_| {
        PowerError::InvalidFormat(
            "release evidence SHA-256 pin must contain lowercase UTF-8 hexadecimal characters"
                .to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_pin_accepts_only_one_exact_digest() {
        let directory = tempfile::tempdir().unwrap();
        let digest = "a".repeat(64);
        for (name, contents) in [
            ("raw", digest.clone()),
            ("lf", format!("{digest}\n")),
            ("crlf", format!("{digest}\r\n")),
        ] {
            let path = directory.path().join(name);
            std::fs::write(&path, contents).unwrap();
            assert_eq!(read_sha256_pin(&path).unwrap(), digest);
        }

        for (name, contents) in [
            ("leading-space", format!(" {digest}")),
            ("trailing-space", format!("{digest} ")),
            ("extra-line", format!("{digest}\n{digest}\n")),
        ] {
            let path = directory.path().join(name);
            std::fs::write(&path, contents).unwrap();
            assert!(read_sha256_pin(&path).is_err());
        }
    }
}
