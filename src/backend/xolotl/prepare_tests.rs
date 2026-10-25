use super::*;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn fixture(root: &Path) -> Assembly {
    let data = [b"GGUF-first".as_slice(), b"second".as_slice()];
    let mut combined = Vec::new();
    let mut parts = Vec::new();
    for (i, bytes) in data.iter().enumerate() {
        let path = root.join(format!("input-{i}"));
        std::fs::write(&path, bytes).unwrap();
        parts.push(XolotlModelPart {
            path,
            size_bytes: bytes.len() as u64,
            sha256: sha(bytes),
        });
        combined.extend_from_slice(bytes);
    }
    Assembly {
        name: "exact-model".into(),
        path: root.join("output/V41-XQ.gguf"),
        expected: sha(&combined),
        size: combined.len() as u64,
        parts,
        deadline: Instant::now() + Duration::from_secs(10),
        cancel: Arc::new(AtomicBool::new(false)),
    }
}
#[test]
fn xolotl_assembly_verifies_every_member_and_the_joined_digest_before_publication() {
    let root = tempfile::tempdir().unwrap();
    let assembly = fixture(root.path());
    let manifest = assembly.prepare().unwrap();
    assert_eq!(manifest.path, assembly.path);
    assert_eq!(manifest.sha256, assembly.expected);
    assert_eq!(std::fs::read(&assembly.path).unwrap(), b"GGUF-firstsecond");
    assert_eq!(
        std::fs::read_dir(assembly.path.parent().unwrap())
            .unwrap()
            .count(),
        1
    );
}
#[test]
fn xolotl_assembly_reuse_is_measured_and_does_not_require_refetching_sources() {
    let root = tempfile::tempdir().unwrap();
    let assembly = fixture(root.path());
    assembly.prepare().unwrap();
    for part in &assembly.parts {
        std::fs::remove_file(&part.path).unwrap();
    }
    assert_eq!(assembly.prepare().unwrap().sha256, assembly.expected);
    std::fs::write(&assembly.path, b"GGUF-evil-second").unwrap();
    assert!(assembly.prepare().is_err());
    assert_eq!(std::fs::read(&assembly.path).unwrap(), b"GGUF-evil-second");
}
#[test]
fn xolotl_assembly_never_publishes_corrupt_changed_or_reordered_members() {
    for corruption in ["contents", "size", "order", "expected"] {
        let root = tempfile::tempdir().unwrap();
        let mut assembly = fixture(root.path());
        match corruption {
            "contents" => std::fs::write(&assembly.parts[0].path, b"GGUF-wrong").unwrap(),
            "size" => assembly.parts[0].size_bytes += 1,
            "order" => assembly.parts.reverse(),
            "expected" => assembly.expected = "1".repeat(64),
            _ => unreachable!(),
        }
        assert!(assembly.prepare().is_err(), "{corruption}");
        assert!(!assembly.path.exists());
        let parent = assembly.path.parent().unwrap();
        if parent.exists() {
            assert_eq!(std::fs::read_dir(parent).unwrap().count(), 0);
        }
    }
}
#[test]
fn xolotl_assembly_refuses_symlinks_without_touching_their_targets() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let assembly = fixture(root.path());
    let target = root.path().join("original");
    std::fs::rename(&assembly.parts[0].path, &target).unwrap();
    symlink(&target, &assembly.parts[0].path).unwrap();
    let original = std::fs::read(&target).unwrap();
    assert!(assembly.prepare().is_err());
    assert_eq!(std::fs::read(&target).unwrap(), original);
}
#[test]
fn xolotl_assembly_honors_cancellation_and_deadline_before_creating_output() {
    for cancelled in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut assembly = fixture(root.path());
        if cancelled {
            assembly.cancel.store(true, Ordering::Release);
        } else {
            assembly.deadline = Instant::now() - Duration::from_secs(1);
        }
        let error = assembly.prepare().unwrap_err().to_string();
        assert!(error.contains(if cancelled { "cancelled" } else { "deadline" }));
        assert!(!assembly.path.parent().unwrap().exists());
    }
}
#[tokio::test]
async fn xolotl_assembly_admission_rejects_oversized_or_duplicate_records() {
    let root = tempfile::tempdir().unwrap();
    let assembly = fixture(root.path());
    for oversized in [false, true] {
        let mut config = PowerConfig::default();
        config.xolotl.model = Some(assembly.path.to_string_lossy().into_owned());
        let parts = if oversized {
            vec![XolotlModelPart {
                path: assembly.parts[0].path.clone(),
                size_bytes: MAX_BYTES + 1,
                sha256: assembly.parts[0].sha256.clone(),
            }]
        } else {
            vec![assembly.parts[0].clone(), assembly.parts[0].clone()]
        };
        config.xolotl.managed = Some(crate::config::XolotlManagedConfig {
            model_name: assembly.name.clone(),
            model_sha256: assembly.expected.clone(),
            model_parts: parts,
            gpu_uuids: vec!["GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".into()],
            python: "/usr/bin/python3.12".into(),
            bridge: "/opt/a3s/bin/bridge.py".into(),
            server: "/opt/xolotl/server.py".into(),
            tokenizer: "/models/tokenizer".into(),
            startup_timeout_seconds: 30,
            shutdown_timeout_seconds: 5,
        });
        assert!(prepare_managed_model(&config, None).await.is_err());
        assert!(!assembly.path.exists());
    }
}

#[test]
fn dspark_binding_refuses_corruption_missing_magic_and_symlink() {
    for bad in ["none", "digest", "magic", "symlink"] {
        let root = tempfile::tempdir().unwrap();
        let a = fixture(root.path());
        let path = root.path().join("draft.gguf");
        let bytes = b"GGUF-owned-draft";
        std::fs::write(&path, bytes).unwrap();
        let mut d = crate::config::XolotlDSparkConfig {
            sidecar: path.clone(),
            sidecar_sha256: sha(bytes),
            depth: 4,
            confidence_bp: 6000,
            evict_prompt_tokens: 4096,
        };
        match bad {
            "digest" => d.sidecar_sha256 = "f".repeat(64),
            "magic" => {
                let b = b"NOPE-owned-draft";
                std::fs::write(&path, b).unwrap();
                d.sidecar_sha256 = sha(b);
            }
            "symlink" => {
                let target = root.path().join("original");
                std::fs::rename(&path, &target).unwrap();
                std::os::unix::fs::symlink(target, &path).unwrap();
            }
            "none" => {}
            _ => unreachable!(),
        }
        assert_eq!(verify_dspark_sidecar(d, &a).is_ok(), bad == "none", "{bad}");
        assert!(!a.path.exists());
    }
}
