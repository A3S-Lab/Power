#![cfg(feature = "embedded-inference")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use sha2::{Digest, Sha256};

const POWER_COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
}

fn build_handoff(root: &Path, manifest: &Path, platform: &str) -> Output {
    command()
        .arg("build-release-handoff")
        .arg("--root")
        .arg(root)
        .arg("--platform")
        .arg(platform)
        .arg("--power-version")
        .arg(env!("CARGO_PKG_VERSION"))
        .arg("--power-commit")
        .arg(POWER_COMMIT)
        .arg("--output")
        .arg(manifest)
        .output()
        .expect("release handoff builder should start")
}

fn verify_handoff(root: &Path, manifest: &Path, platform: &str) -> Output {
    command()
        .arg("verify-release-handoff")
        .arg("--manifest")
        .arg(manifest)
        .arg("--root")
        .arg(root)
        .arg("--platform")
        .arg(platform)
        .arg("--power-version")
        .arg(env!("CARGO_PKG_VERSION"))
        .arg("--power-commit")
        .arg(POWER_COMMIT)
        .output()
        .expect("release handoff verifier should start")
}

fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let directory = tempfile::tempdir().expect("temporary directory should exist");
    let root = directory.path().join("metal-handoff");
    std::fs::create_dir_all(root.join("hardware")).unwrap();
    std::fs::write(root.join("build.log"), b"cargo build --locked\n").unwrap();
    std::fs::write(root.join("hardware/apple.txt"), b"Apple M4 Max\n").unwrap();
    std::fs::write(root.join("metal.json"), b"{\"capture\":true}\n").unwrap();
    let manifest = directory.path().join("metal-handoff.manifest.json");
    (directory, root, manifest)
}

#[test]
fn handoff_builder_binds_the_exact_portable_directory_inventory() {
    let (_directory, root, manifest_path) = fixture();
    let output = build_handoff(&root, &manifest_path, "metal");
    assert!(
        output.status.success(),
        "handoff build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["schema"], "a3s.power.release-handoff-build.v1");
    assert_eq!(receipt["built"], true);
    assert_eq!(receipt["exactRootInventory"], true);
    assert_eq!(receipt["strictV1BundleRequired"], true);
    assert_eq!(receipt["fileCount"], 3);

    let encoded = std::fs::read(&manifest_path).unwrap();
    assert_eq!(
        receipt["manifestSha256"],
        format!("{:x}", Sha256::digest(&encoded))
    );
    let manifest: Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(manifest["schema"], "a3s.power.release-handoff.v1");
    assert_eq!(manifest["platform"], "metal");
    assert_eq!(manifest["powerVersion"], env!("CARGO_PKG_VERSION"));
    assert_eq!(manifest["powerCommit"], POWER_COMMIT);
    let paths = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["path"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["build.log", "hardware/apple.txt", "metal.json"]);
    assert!(paths
        .iter()
        .all(|path| !Path::new(path).is_absolute() && !path.contains('\\')));

    let verified = verify_handoff(&root, &manifest_path, "metal");
    assert!(
        verified.status.success(),
        "handoff verification failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let verification: Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(
        verification["schema"],
        "a3s.power.release-handoff-verification.v1"
    );
    assert_eq!(verification["verified"], true);
    assert_eq!(verification["manifestSha256"], receipt["manifestSha256"]);
    assert_eq!(verification["fileCount"], 3);
}

#[test]
fn handoff_verifier_rejects_mutation_missing_and_extra_files() {
    let (_directory, root, manifest_path) = fixture();
    let built = build_handoff(&root, &manifest_path, "metal");
    assert!(built.status.success());

    std::fs::write(root.join("build.log"), b"changed build\n").unwrap();
    let mutated = verify_handoff(&root, &manifest_path, "metal");
    assert!(!mutated.status.success());
    assert!(String::from_utf8_lossy(&mutated.stderr).contains("inventory differs"));

    std::fs::write(root.join("build.log"), b"cargo build --locked\n").unwrap();
    std::fs::remove_file(root.join("metal.json")).unwrap();
    let missing = verify_handoff(&root, &manifest_path, "metal");
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("inventory differs"));

    std::fs::write(root.join("metal.json"), b"{\"capture\":true}\n").unwrap();
    std::fs::write(root.join("unreviewed.tmp"), b"unexpected").unwrap();
    let extra = verify_handoff(&root, &manifest_path, "metal");
    assert!(!extra.status.success());
    assert!(String::from_utf8_lossy(&extra.stderr).contains("inventory differs"));
}

#[test]
fn handoff_verifier_rejects_relabeling_and_unsafe_manifest_paths() {
    let (_directory, root, manifest_path) = fixture();
    let built = build_handoff(&root, &manifest_path, "metal");
    assert!(built.status.success());

    let wrong_platform = verify_handoff(&root, &manifest_path, "cuda");
    assert!(!wrong_platform.status.success());
    assert!(String::from_utf8_lossy(&wrong_platform.stderr)
        .contains("does not match the expected platform, version, and source revision"));

    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let original = manifest.clone();
    manifest["files"][0]["path"] = Value::String("../outside".to_string());
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let traversal = verify_handoff(&root, &manifest_path, "metal");
    assert!(!traversal.status.success());
    assert!(String::from_utf8_lossy(&traversal.stderr).contains("portable relative path"));

    let mut unknown = original;
    unknown["unexpected"] = Value::Bool(true);
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&unknown).unwrap()).unwrap();
    let unknown_field = verify_handoff(&root, &manifest_path, "metal");
    assert!(!unknown_field.status.success());
    assert!(String::from_utf8_lossy(&unknown_field.stderr).contains("unknown field"));
}

#[test]
fn handoff_manifest_must_be_outside_the_staged_artifact_root() {
    let (_directory, root, _manifest_path) = fixture();
    let inside = root.join("manifest.json");
    let output = build_handoff(&root, &inside, "metal");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("manifest must be stored outside the staged artifact root"));
    assert!(!inside.exists());
}

#[cfg(unix)]
#[test]
fn handoff_builder_rejects_symbolic_links() {
    use std::os::unix::fs::symlink;

    let (_directory, root, manifest_path) = fixture();
    symlink(root.join("metal.json"), root.join("capture-link.json")).unwrap();
    let output = build_handoff(&root, &manifest_path, "metal");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("symbolic links"));
    assert!(!manifest_path.exists());
}
