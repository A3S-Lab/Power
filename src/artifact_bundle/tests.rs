use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::*;

struct OneShotServer {
    url: String,
    calls: Arc<AtomicUsize>,
    worker: Option<JoinHandle<()>>,
}

impl OneShotServer {
    fn start(body: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let worker_calls = Arc::clone(&calls);
        let worker = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("test artifact server failed: {error}"),
                }
            };
            worker_calls.fetch_add(1, Ordering::SeqCst);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            let response_header = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            if response_header.is_ok() {
                let _ = stream.write_all(&body);
            }
        });
        Self {
            url: format!("http://{address}/artifact"),
            calls,
            worker: Some(worker),
        }
    }

    fn finish(mut self) -> usize {
        self.worker.take().unwrap().join().unwrap();
        self.calls.load(Ordering::SeqCst)
    }
}

impl Drop for OneShotServer {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

fn inline_bundle(bytes: &[u8]) -> ArtifactBundle {
    let digest = sha256_bytes(bytes);
    ArtifactBundle::new(
        "test/inline",
        "revision-1",
        vec![BundleArtifact::inline("model.bin", bytes.to_vec(), digest).unwrap()],
    )
    .unwrap()
}

fn remote_bundle(url: &str, bytes: &[u8]) -> ArtifactBundle {
    ArtifactBundle::new(
        "test/remote",
        "revision-1",
        vec![BundleArtifact::remote(
            "model.bin",
            url,
            sha256_bytes(bytes),
            u64::try_from(bytes.len()).unwrap(),
        )
        .unwrap()],
    )
    .unwrap()
}

#[test]
fn public_bundle_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ArtifactBundle>();
    assert_send_sync::<BundleArtifact>();
    assert_send_sync::<BundleProvisionPolicy>();
    assert_send_sync::<ProvisionedArtifactBundle>();
    assert_send_sync::<ArtifactBundleError>();
}

#[test]
fn artifact_spec_rejects_paths_credentials_and_bad_digests() {
    for name in ["../model", "nested/model", "nested\\model", RECEIPT_NAME] {
        assert!(BundleArtifact::inline(name, b"x".to_vec(), sha256_bytes(b"x")).is_err());
    }
    assert!(BundleArtifact::remote(
        "model.bin",
        "https://user:secret@example.test/model",
        sha256_bytes(b"x"),
        1,
    )
    .is_err());
    assert!(BundleArtifact::remote(
        "model.bin",
        "http://example.test/model",
        sha256_bytes(b"x"),
        1,
    )
    .is_err());
    assert!(BundleArtifact::remote("model.bin", "https://example.test/model", "ABC", 1,).is_err());
}

#[test]
fn debug_output_omits_remote_urls_and_inline_bytes() {
    let remote_url = "https://example.test/private/model";
    let remote =
        BundleArtifact::remote("remote.bin", remote_url, sha256_bytes(b"remote"), 64).unwrap();
    let inline = BundleArtifact::inline(
        "inline.bin",
        b"inline-secret-marker".to_vec(),
        sha256_bytes(b"inline-secret-marker"),
    )
    .unwrap();
    let rendered = format!("{remote:?} {inline:?}");
    assert!(!rendered.contains(remote_url));
    assert!(!rendered.contains("inline-secret-marker"));
    assert!(rendered.contains("remote"));
    assert!(rendered.contains("inline"));
}

#[tokio::test]
async fn inline_bundle_materializes_and_reuses_without_network() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("bundle");
    let bundle = inline_bundle(b"trusted-inline-model");
    let policy = BundleProvisionPolicy::new(&destination).with_network(false);

    let first = provision_artifact_bundle(&bundle, &policy).await.unwrap();
    assert_eq!(first.installed_artifacts(), 1);
    assert_eq!(first.reused_artifacts(), 0);
    assert_eq!(
        std::fs::read(destination.join("model.bin")).unwrap(),
        b"trusted-inline-model"
    );

    let second = provision_artifact_bundle(&bundle, &policy).await.unwrap();
    assert_eq!(second.installed_artifacts(), 0);
    assert_eq!(second.reused_artifacts(), 1);
    let receipt = std::fs::read_to_string(destination.join(RECEIPT_NAME)).unwrap();
    assert!(receipt.contains("a3s.power.artifact-bundle.v1"));
    assert!(receipt.contains(bundle.name()));
    assert!(receipt.contains(bundle.revision()));
}

#[tokio::test]
async fn missing_remote_artifact_fails_closed_offline() {
    let temp = tempfile::tempdir().unwrap();
    let artifact = BundleArtifact::remote(
        "model.bin",
        "https://example.test/model.bin",
        sha256_bytes(b"model"),
        5,
    )
    .unwrap();
    let bundle = ArtifactBundle::new("test/offline", "revision-1", vec![artifact]).unwrap();
    let error = provision_artifact_bundle(
        &bundle,
        &BundleProvisionPolicy::new(temp.path().join("bundle")).with_network(false),
    )
    .await
    .unwrap_err();

    assert!(matches!(error, ArtifactBundleError::OfflineMissing { .. }));
    assert!(!error.to_string().contains("example.test"));
}

#[tokio::test]
async fn concurrent_first_use_downloads_once_then_reuses_offline() {
    let bytes = b"remote-model-bytes".to_vec();
    let server = OneShotServer::start(bytes.clone());
    let bundle = remote_bundle(&server.url, &bytes);
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("bundle");
    let policy = BundleProvisionPolicy::new(&destination);

    let (first, second) = tokio::join!(
        provision_artifact_bundle(&bundle, &policy),
        provision_artifact_bundle(&bundle, &policy)
    );
    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(
        first.installed_artifacts() + second.installed_artifacts(),
        1
    );
    assert_eq!(first.reused_artifacts() + second.reused_artifacts(), 1);
    assert_eq!(server.finish(), 1);
    assert_eq!(std::fs::read(destination.join("model.bin")).unwrap(), bytes);

    let receipt = std::fs::read_to_string(destination.join(RECEIPT_NAME)).unwrap();
    assert!(!receipt.contains("127.0.0.1"));
    let offline = BundleProvisionPolicy::new(&destination).with_network(false);
    let reused = provision_artifact_bundle(&bundle, &offline).await.unwrap();
    assert_eq!(reused.installed_artifacts(), 0);
    assert_eq!(reused.reused_artifacts(), 1);
}

#[tokio::test]
async fn digest_failure_never_commits_downloaded_bytes() {
    let server = OneShotServer::start(b"replaced".to_vec());
    let bundle = remote_bundle(&server.url, b"expected");
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("bundle");
    let error = provision_artifact_bundle(&bundle, &BundleProvisionPolicy::new(&destination))
        .await
        .unwrap_err();

    assert!(matches!(error, ArtifactBundleError::Integrity { .. }));
    assert!(!destination.join("model.bin").exists());
    let partials = std::fs::read_dir(&destination)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".partial"))
        .count();
    assert_eq!(partials, 0);
    assert_eq!(server.finish(), 1);
}

#[tokio::test]
async fn oversized_download_never_commits_bytes() {
    let bytes = b"larger-than-policy".to_vec();
    let server = OneShotServer::start(bytes.clone());
    let artifact = BundleArtifact::remote(
        "model.bin",
        &server.url,
        sha256_bytes(&bytes),
        u64::try_from(bytes.len() - 1).unwrap(),
    )
    .unwrap();
    let bundle = ArtifactBundle::new("test/oversized", "revision-1", vec![artifact]).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("bundle");

    let error = provision_artifact_bundle(&bundle, &BundleProvisionPolicy::new(&destination))
        .await
        .unwrap_err();

    assert!(matches!(error, ArtifactBundleError::TooLarge { .. }));
    assert!(!destination.join("model.bin").exists());
    assert_eq!(server.finish(), 1);
}

#[tokio::test]
async fn installed_digest_mismatch_is_not_silently_replaced() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("bundle");
    let bundle = inline_bundle(b"trusted");
    let policy = BundleProvisionPolicy::new(&destination).with_network(false);
    provision_artifact_bundle(&bundle, &policy).await.unwrap();
    std::fs::write(destination.join("model.bin"), b"changed").unwrap();

    let error = provision_artifact_bundle(&bundle, &policy)
        .await
        .unwrap_err();
    assert!(matches!(error, ArtifactBundleError::Integrity { .. }));
    assert_eq!(
        std::fs::read(destination.join("model.bin")).unwrap(),
        b"changed"
    );
}
