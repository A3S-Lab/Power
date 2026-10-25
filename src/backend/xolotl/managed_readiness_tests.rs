#![cfg(test)]

use super::*;
use crate::backend::{xolotl::XolotlBackend, Backend};
use crate::config::PowerConfig;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;

fn backend(root: &Path, engine: Option<ManagedEngine>) -> XolotlBackend {
    let mut config = PowerConfig::default();
    config.xolotl.managed = Some(settings(root));
    let mut backend = XolotlBackend::new(Arc::new(config));
    *backend.engine.get_mut() = engine;
    backend
}

#[tokio::test]
async fn concurrent_readiness_checks_wait_for_the_healthy_owned_probe() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let entered_server = entered.clone();
    let release_server = release.clone();
    let server = tokio::spawn(async move {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            stream.read(&mut request).await.unwrap();
            if index == 0 {
                entered_server.notify_one();
                release_server.notified().await;
            }
            let body = r#"{"loaded":true,"model":"exact-model","service":"strata"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let root = tempfile::tempdir().unwrap();
    let mut engine = process("exec sleep 60");
    engine.origin = format!("http://127.0.0.1:{port}");
    let backend = Arc::new(backend(root.path(), Some(engine)));
    let first_backend = backend.clone();
    let first = tokio::spawn(async move { first_backend.runtime_readiness().await });
    tokio::time::timeout(Duration::from_secs(1), entered.notified())
        .await
        .unwrap();
    let second_backend = backend.clone();
    let mut second = tokio::spawn(async move { second_backend.runtime_readiness().await });
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut second)
            .await
            .is_err(),
        "a concurrent probe must wait, not report an already-loaded engine unavailable"
    );
    release.notify_one();
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    backend
        .engine
        .lock()
        .await
        .as_mut()
        .unwrap()
        .stop()
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn readiness_still_refuses_a_lifecycle_lock_after_its_bounded_deadline() {
    let root = tempfile::tempdir().unwrap();
    let backend = backend(root.path(), None);
    let _lifecycle = backend.engine.lock().await;
    let started = tokio::time::Instant::now();
    let error = backend.runtime_readiness().await.unwrap_err().to_string();
    assert!(error.contains("readiness deadline exceeded"), "{error}");
    assert_eq!(started.elapsed(), Duration::from_secs(2));
}
