#![cfg(feature = "server")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration as StdDuration, Instant};

use a3s_power::api::distributed_serving::{
    AbortDistributedExecutionResponse, DistributedDecodeStreamEvent, DistributedDecodeStreamFrame,
    DistributedPhaseDecision, DistributedPhaseResponse, DistributedProtocolErrorCode,
    DistributedProtocolErrorResponse, DistributedResponseChunk, PreparedDecodeResult,
    PublishedPrefillResult, DISTRIBUTED_SERVING_SCHEMA, DISTRIBUTED_SERVING_STREAM_SCHEMA,
};
use a3s_power::serving::DisaggregatedServingRole;
use chrono::{Duration, Utc};
use reqwest::{Client, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use tempfile::TempDir;
use uuid::Uuid;

#[path = "distributed_serving_cross_process/fixture.rs"]
mod fixture;

use fixture::{ReadyWorker, MODEL, SERVICE_KEY};

const ROLE_ENV: &str = "A3S_POWER_CONFORMANCE_ROLE";
const READY_ENV: &str = "A3S_POWER_CONFORMANCE_READY";
const SHUTDOWN_ENV: &str = "A3S_POWER_CONFORMANCE_SHUTDOWN";

struct WorkerProcess {
    child: Option<Child>,
    ready_path: PathBuf,
    shutdown_path: PathBuf,
}

impl WorkerProcess {
    fn spawn(directory: &Path, name: &str, role: DisaggregatedServingRole) -> Self {
        let ready_path = directory.join(format!("{name}.ready.json"));
        let shutdown_path = directory.join(format!("{name}.shutdown"));
        let role = match role {
            DisaggregatedServingRole::Prefill => "prefill",
            DisaggregatedServingRole::Decode => "decode",
        };
        let child = Command::new(std::env::current_exe().expect("test executable is available"))
            .args([
                "--ignored",
                "--exact",
                "cross_process_worker",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ROLE_ENV, role)
            .env(READY_ENV, &ready_path)
            .env(SHUTDOWN_ENV, &shutdown_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("cross-process worker starts");
        Self {
            child: Some(child),
            ready_path,
            shutdown_path,
        }
    }

    async fn wait_until_ready(&mut self) -> ReadyWorker {
        let deadline = Instant::now() + StdDuration::from_secs(15);
        loop {
            if let Ok(document) = std::fs::read(&self.ready_path) {
                return serde_json::from_slice(&document).expect("worker readiness is valid JSON");
            }
            let child = self.child.as_mut().expect("worker is still owned");
            if let Some(status) = child.try_wait().expect("worker status is observable") {
                panic!("cross-process worker exited before readiness: {status}");
            }
            assert!(
                Instant::now() < deadline,
                "cross-process worker did not become ready"
            );
            tokio::time::sleep(StdDuration::from_millis(25)).await;
        }
    }

    async fn stop(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        std::fs::write(&self.shutdown_path, b"shutdown")
            .expect("worker shutdown signal is written");
        let deadline = Instant::now() + StdDuration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().expect("worker status is observable") {
                assert!(status.success(), "cross-process worker failed: {status}");
                self.child = None;
                return;
            }
            if Instant::now() >= deadline {
                child.kill().expect("timed-out worker can be killed");
                let status = child.wait().expect("killed worker can be reaped");
                self.child = None;
                panic!("cross-process worker did not stop gracefully: {status}");
            }
            tokio::time::sleep(StdDuration::from_millis(25)).await;
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn completion_payload() -> Value {
    json!({
        "endpoint": "completions",
        "body": {
            "model": MODEL,
            "prompt": "private cross-process prompt",
            "stream": true
        }
    })
}

async fn post(client: &Client, worker: &ReadyWorker, path: &str, body: Value) -> Response {
    client
        .post(format!("http://{}{}", worker.address, path))
        .bearer_auth(SERVICE_KEY)
        .json(&body)
        .send()
        .await
        .expect("worker HTTP request succeeds")
}

async fn json_body<T: DeserializeOwned>(response: Response) -> T {
    response
        .json::<T>()
        .await
        .expect("worker response is valid JSON")
}

async fn prepare_decode(
    client: &Client,
    decode: &ReadyWorker,
    execution_id: Uuid,
    expires_at: chrono::DateTime<Utc>,
) -> a3s_power::serving::StateTransferTarget {
    let response = post(
        client,
        decode,
        "/internal/v1/distributed-serving/decode/prepare",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.worker_epoch,
            "execution_profile_sha256": decode.execution_profile_sha256,
            "expires_at": expires_at,
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let response: DistributedPhaseResponse<PreparedDecodeResult> = json_body(response).await;
    match response.outcome {
        DistributedPhaseDecision::Ready { result } => result.target,
        outcome => panic!("decode preparation was not ready: {outcome:?}"),
    }
}

async fn publish_prefill(
    client: &Client,
    prefill: &ReadyWorker,
    execution_id: Uuid,
    expires_at: chrono::DateTime<Utc>,
    target: a3s_power::serving::StateTransferTarget,
) -> a3s_power::serving::StateTransferSource {
    let response = post(
        client,
        prefill,
        "/internal/v1/distributed-serving/prefill/execute",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": prefill.worker_epoch,
            "execution_profile_sha256": prefill.execution_profile_sha256,
            "expires_at": expires_at,
            "request": completion_payload(),
            "target": target
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let response: DistributedPhaseResponse<PublishedPrefillResult> = json_body(response).await;
    match response.outcome {
        DistributedPhaseDecision::Ready { result } => result.source,
        outcome => panic!("prefill execution was not ready: {outcome:?}"),
    }
}

async fn abort_execution(client: &Client, worker: &ReadyWorker, execution_id: Uuid) {
    let response = post(
        client,
        worker,
        "/internal/v1/distributed-serving/abort",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": worker.worker_epoch,
            "execution_profile_sha256": worker.execution_profile_sha256
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let response: AbortDistributedExecutionResponse = json_body(response).await;
    assert!(response.accepted);
}

#[tokio::test]
async fn distributed_serving_crosses_processes_and_fails_closed_on_peer_loss_and_restart() {
    let directory = TempDir::new().expect("temporary process directory is available");
    let mut decode_process = WorkerProcess::spawn(
        directory.path(),
        "decode-1",
        DisaggregatedServingRole::Decode,
    );
    let mut prefill_process = WorkerProcess::spawn(
        directory.path(),
        "prefill-1",
        DisaggregatedServingRole::Prefill,
    );
    let decode = decode_process.wait_until_ready().await;
    let prefill = prefill_process.wait_until_ready().await;
    assert_ne!(decode.process_id, prefill.process_id);
    assert_ne!(decode.process_id, std::process::id());
    assert_ne!(prefill.process_id, std::process::id());
    let client = Client::builder()
        .timeout(StdDuration::from_secs(12))
        .build()
        .expect("test HTTP client is valid");

    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::seconds(10);
    let target = prepare_decode(&client, &decode, execution_id, expires_at).await;
    let source = publish_prefill(&client, &prefill, execution_id, expires_at, target).await;
    let response = post(
        &client,
        &decode,
        "/internal/v1/distributed-serving/decode/execute",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": execution_id,
            "worker_epoch": decode.worker_epoch,
            "execution_profile_sha256": decode.execution_profile_sha256,
            "source": source
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[reqwest::header::CONTENT_TYPE],
        "application/x-ndjson"
    );
    let body = response.bytes().await.expect("decode stream is readable");
    let frames = body
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_slice::<DistributedDecodeStreamFrame>(line)
                .expect("decode stream frame is valid")
        })
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 3);
    assert!(frames.iter().all(|frame| {
        frame.schema == DISTRIBUTED_SERVING_STREAM_SCHEMA
            && frame.execution_id == execution_id
            && frame.worker_epoch == decode.worker_epoch
    }));
    assert!(matches!(
        frames[0].payload,
        DistributedDecodeStreamEvent::Ready
    ));
    match &frames[1].payload {
        DistributedDecodeStreamEvent::Chunk {
            sequence: 0,
            response: DistributedResponseChunk::Completions(chunk),
        } => assert_eq!(chunk.text, "cross-process-token"),
        event => panic!("unexpected decode frame: {event:?}"),
    }
    assert!(matches!(
        frames[2].payload,
        DistributedDecodeStreamEvent::Completed { sequence: 1 }
    ));
    abort_execution(&client, &prefill, execution_id).await;
    abort_execution(&client, &decode, execution_id).await;

    let lost_peer_execution = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::seconds(10);
    let target = prepare_decode(&client, &decode, lost_peer_execution, expires_at).await;
    let source = publish_prefill(&client, &prefill, lost_peer_execution, expires_at, target).await;
    prefill_process.stop().await;
    let response = post(
        &client,
        &decode,
        "/internal/v1/distributed-serving/decode/execute",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": lost_peer_execution,
            "worker_epoch": decode.worker_epoch,
            "execution_profile_sha256": decode.execution_profile_sha256,
            "source": source
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error: DistributedProtocolErrorResponse = json_body(response).await;
    assert_eq!(error.code, DistributedProtocolErrorCode::Unavailable);

    decode_process.stop().await;
    let mut restarted_decode_process = WorkerProcess::spawn(
        directory.path(),
        "decode-2",
        DisaggregatedServingRole::Decode,
    );
    let restarted_decode = restarted_decode_process.wait_until_ready().await;
    assert_ne!(restarted_decode.process_id, decode.process_id);
    assert_ne!(restarted_decode.worker_epoch, decode.worker_epoch);
    assert_eq!(
        restarted_decode.execution_profile_sha256,
        decode.execution_profile_sha256
    );
    let response = post(
        &client,
        &restarted_decode,
        "/internal/v1/distributed-serving/decode/prepare",
        json!({
            "schema": DISTRIBUTED_SERVING_SCHEMA,
            "execution_id": Uuid::new_v4(),
            "worker_epoch": decode.worker_epoch,
            "execution_profile_sha256": restarted_decode.execution_profile_sha256,
            "expires_at": Utc::now() + Duration::seconds(10),
            "request": completion_payload()
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: DistributedProtocolErrorResponse = json_body(response).await;
    assert_eq!(error.code, DistributedProtocolErrorCode::StaleWorker);
    restarted_decode_process.stop().await;
}

#[tokio::test]
#[ignore = "spawned only by the cross-process conformance parent"]
async fn cross_process_worker() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let role = match role.as_str() {
        "prefill" => DisaggregatedServingRole::Prefill,
        "decode" => DisaggregatedServingRole::Decode,
        value => panic!("unknown cross-process worker role: {value}"),
    };
    let ready_path = PathBuf::from(std::env::var_os(READY_ENV).expect("ready path is provided"));
    let shutdown_path =
        PathBuf::from(std::env::var_os(SHUTDOWN_ENV).expect("shutdown path is provided"));
    fixture::serve_worker(role, &ready_path, &shutdown_path)
        .await
        .expect("cross-process worker serves successfully");
}
