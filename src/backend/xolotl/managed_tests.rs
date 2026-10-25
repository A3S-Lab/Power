use super::*;
use std::path::Path;

fn settings(root: &Path) -> XolotlManagedConfig {
    for n in ["python", "bridge.py", "server.py"] {
        std::fs::write(root.join(n), b"fixture").unwrap();
    }
    std::fs::create_dir(root.join("tokenizer")).unwrap();
    XolotlManagedConfig {
        model_sha256: "1".repeat(64),
        model_parts: Vec::new(),
        gpu_uuids: vec!["GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".into()],
        model_name: "exact-model".into(),
        python: root.join("python"),
        bridge: root.join("bridge.py"),
        server: root.join("server.py"),
        tokenizer: root.join("tokenizer"),
        startup_timeout_seconds: 10,
        shutdown_timeout_seconds: 1,
    }
}
#[test]
fn managed_origin_identity_and_deadlines_are_closed() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = settings(root.path());
    assert_eq!(
        validate_config(&cfg, "http://127.0.0.1:8150", "exact-model").unwrap(),
        8150
    );
    for origin in [
        "https://127.0.0.1:8150",
        "http://localhost:8150",
        "http://0.0.0.0:8150",
        "http://user:secret@127.0.0.1:8150",
        "http://127.0.0.1:8150/v1",
        "http://127.0.0.1:8150?token=secret",
    ] {
        let error = validate_config(&cfg, origin, "exact-model")
            .unwrap_err()
            .to_string();
        assert!(!error.contains("secret"));
    }
    assert!(validate_config(&cfg, "http://127.0.0.1:8150", "another-model").is_err());
    cfg.startup_timeout_seconds = 3601;
    assert!(validate_config(&cfg, "http://127.0.0.1:8150", "exact-model").is_err());
    cfg.startup_timeout_seconds = 10;
    cfg.shutdown_timeout_seconds = 61;
    assert!(validate_config(&cfg, "http://127.0.0.1:8150", "exact-model").is_err());
}
#[test]
fn managed_ready_proof_refuses_unknown_facts_and_payloads() {
    assert!(serde_json::from_slice::<Ready>(br#"{"schema":"a3s.power.xolotl.ready.v2","nonce":"n","model_name":"m","engine_pid":123,"mock":true}"#).is_err());
    assert!(serde_json::from_slice::<Ready>(
        br#"{"schema":"a3s.power.xolotl.ready.v2","nonce":"n","model_name":"m","engine_pid":true}"#
    )
    .is_err());
}
#[cfg(target_os = "linux")]
fn process(script: &str) -> ManagedEngine {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", script])
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.as_std_mut().process_group(0);
    let child = cmd.spawn().unwrap();
    let group = child.id().unwrap() as i32;
    ManagedEngine {
        child,
        group,
        engine_pid: group,
        model_name: "exact-model".into(),
        origin: "http://127.0.0.1:1".into(),
        shutdown: Duration::from_millis(50),
    }
}
#[cfg(target_os = "linux")]
#[tokio::test]
async fn managed_shutdown_kills_and_reaps_a_term_resistant_owned_group() {
    let mut engine = process("trap '' TERM; exec sleep 60");
    tokio::time::sleep(Duration::from_millis(30)).await;
    engine.ensure_alive().unwrap();
    let pid = engine.group;
    engine.stop().await.unwrap();
    assert!(engine.child.try_wait().unwrap().is_some());
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
}
#[cfg(target_os = "linux")]
#[tokio::test]
async fn managed_probe_refuses_an_http_200_for_the_wrong_or_unloaded_model() {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };
    for body in [
        r#"{"loaded":true,"model":"other","service":"strata"}"#,
        r#"{"loaded":false,"model":"exact-model","service":"strata"}"#,
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let serve = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            stream.read(&mut request).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let mut engine = process("exec sleep 60");
        engine.origin = format!("http://127.0.0.1:{port}");
        let error = engine
            .probe(&reqwest::Client::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not prove this model is loaded"));
        engine.stop().await.unwrap();
        serve.await.unwrap();
    }
}

#[test]
fn managed_local_index_must_resolve_to_the_exact_cloud_admitted_uuid() {
    let selected = "GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".to_string();
    let other = "GPU-aeb13ea2-62f0-f454-47a2-d4cabe61b14d".to_string();
    let row = format!("0, {selected}");
    assert_eq!(
        bind_device_identities(&[0], &[selected.clone()], &row).unwrap(),
        selected
    );
    for csv in [
        format!("0, {other}"),
        format!("0, {selected}\n0, {selected}"),
        format!("0, {selected}\n1, {selected}"),
    ] {
        assert!(bind_device_identities(&[0], &[selected.clone()], &csv).is_err());
    }
    assert!(bind_device_identities(&[1], &[selected], &row).is_err());
}

#[test]
fn managed_context_ready_must_prove_actual_capacity() {
    let mut plan = XolotlLaunchPlan {
        argv: vec!["--ctx".into(), "1048576".into()],
        env: vec![],
        nvidia_indices: vec![0],
        visible_ordinals: vec![0],
        ready_log_line: String::new(),
    };
    validate_context_proof(&plan, 1_048_576).unwrap();
    assert!(validate_context_proof(&plan, 4096).is_err());
    assert!(validate_context_proof(&plan, 1_048_575).is_err());
    plan.argv.pop();
    assert!(validate_context_proof(&plan, 4096).is_err());
}

#[cfg(target_os = "linux")]
#[path = "managed_readiness_tests.rs"]
mod readiness_tests;
