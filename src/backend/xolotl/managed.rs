//! A single owned process group. The adapter must attest the real resident
//! engine's READY handshake and then expose that same model as loaded.
use super::XolotlLaunchPlan;
use crate::{
    config::XolotlManagedConfig,
    error::{PowerError, Result},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
};
const MAX_LAUNCH_BYTES: usize = 64 * 1024;
const MAX_READY_BYTES: u64 = 4096;
const MAX_HEALTH_BYTES: usize = 32 * 1024;

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct Launch<'a> {
    schema: &'static str,
    nonce: &'a str,
    parent_pid: u32,
    server: &'a std::path::Path,
    tokenizer: &'a std::path::Path,
    model_name: &'a str,
    engine_argv: &'a [String],
    engine_env: BTreeMap<&'a str, &'a str>,
    host: &'static str,
    port: u16,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    schema: String,
    nonce: String,
    model_name: String,
    engine_pid: u32,
    context_tokens: u32,
    #[serde(default)]
    dspark_active: bool,
}
pub(super) struct ManagedEngine {
    child: Child,
    group: i32,
    engine_pid: i32,
    model_name: String,
    origin: String,
    shutdown: Duration,
}
fn refused(reason: impl Into<String>) -> PowerError {
    PowerError::InferenceFailed(reason.into())
}

fn validate_context_proof(plan: &XolotlLaunchPlan, actual: u32) -> Result<()> {
    let expected = plan
        .argv
        .iter()
        .position(|arg| arg == "--ctx")
        .map(|i| plan.argv.get(i + 1).and_then(|s| s.parse::<u32>().ok()));
    // Launch plans originate in the typed planner; still reject malformed facts.
    let expected = match expected {
        Some(Some(value)) => value,
        Some(None) => return Err(refused("managed Xolotl context launch fact is invalid")),
        None => 4096,
    };
    if actual != expected {
        return Err(refused(
            "managed Xolotl actual context differs from its admitted capacity",
        ));
    }
    Ok(())
}

fn validate_config(config: &XolotlManagedConfig, origin: &str, model_name: &str) -> Result<u16> {
    if config.model_name != model_name
        || model_name.is_empty()
        || model_name.len() > 256
        || model_name.contains(['\0', '\r', '\n'])
    {
        return Err(refused(
            "managed Xolotl model identity does not match its ACL binding",
        ));
    }
    if !(1..=3600).contains(&config.startup_timeout_seconds)
        || !(1..=60).contains(&config.shutdown_timeout_seconds)
    {
        return Err(refused(
            "managed Xolotl startup/shutdown deadlines exceed their bounds",
        ));
    }
    for path in [&config.python, &config.bridge, &config.server] {
        if !path.is_absolute() || !path.is_file() {
            return Err(refused(
                "managed Xolotl executable or adapter path is not an absolute file",
            ));
        }
    }
    if !config.tokenizer.is_absolute() || !config.tokenizer.is_dir() {
        return Err(refused(
            "managed Xolotl tokenizer is not an absolute directory",
        ));
    }
    let url =
        reqwest::Url::parse(origin).map_err(|_| refused("managed Xolotl origin is invalid"))?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return Err(refused(
            "managed Xolotl origin must be an IPv4 loopback HTTP origin",
        ));
    }
    url.port()
        .filter(|port| *port > 1)
        .ok_or_else(|| refused("managed Xolotl origin requires an explicit port"))
}

fn bind_device_identities(indices: &[u32], expected: &[String], csv: &str) -> Result<String> {
    if indices.len() != expected.len() || !matches!(indices.len(), 1 | 8) || csv.len() > 4096 {
        return Err(refused(
            "managed Xolotl device identity inventory exceeds its bounds",
        ));
    }
    let mut inventory = BTreeMap::new();
    let mut identities = std::collections::BTreeSet::new();
    for line in csv.lines().filter(|line| !line.trim().is_empty()) {
        let (index, identity) = line
            .split_once(',')
            .ok_or_else(|| refused("managed Xolotl device identity row is malformed"))?;
        let index = index
            .trim()
            .parse::<u32>()
            .map_err(|_| refused("managed Xolotl device index is invalid"))?;
        let identity = identity.trim();
        let raw = identity
            .strip_prefix("GPU-")
            .ok_or_else(|| refused("managed Xolotl device identity is not a GPU UUID"))?;
        let parsed = uuid::Uuid::parse_str(raw)
            .map_err(|_| refused("managed Xolotl device UUID is invalid"))?;
        if parsed.to_string() != raw
            || inventory.insert(index, identity).is_some()
            || !identities.insert(identity)
            || inventory.len() > 16
        {
            return Err(refused(
                "managed Xolotl device inventory is duplicated or noncanonical",
            ));
        }
    }
    let mut selected = std::collections::BTreeSet::new();
    for (index, expected) in indices.iter().zip(expected) {
        if inventory.get(index).copied() != Some(expected.as_str()) || !selected.insert(expected) {
            return Err(refused(
                "managed Xolotl GPU UUID does not match the admitted local index",
            ));
        }
    }
    Ok(expected.join(","))
}
async fn query_device_identities() -> Result<String> {
    let mut child = Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| refused("managed Xolotl GPU identity query could not start"))?;
    let output = child
        .stdout
        .take()
        .ok_or_else(|| refused("managed Xolotl GPU query omitted stdout"))?;
    let result = tokio::time::timeout(Duration::from_secs(5), async {
        let mut bytes = Vec::new();
        output
            .take(4097)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| refused("managed Xolotl GPU identity query failed"))?;
        if bytes.len() > 4096 {
            return Err(refused(
                "managed Xolotl GPU identity query exceeds its byte bound",
            ));
        }
        if !child
            .wait()
            .await
            .map_err(|_| refused("managed Xolotl GPU query could not be reaped"))?
            .success()
        {
            return Err(refused("managed Xolotl GPU identity query was refused"));
        }
        String::from_utf8(bytes)
            .map_err(|_| refused("managed Xolotl GPU identity query is not UTF-8"))
    })
    .await
    .map_err(|_| refused("managed Xolotl GPU identity query exceeded its deadline"))?;
    if result.is_err() {
        let _ = child.start_kill();
    }
    result
}

impl ManagedEngine {
    #[cfg(target_os = "linux")]
    pub(super) async fn start(
        config: &XolotlManagedConfig,
        plan: &XolotlLaunchPlan,
        origin: &str,
        model_name: &str,
        http: &reqwest::Client,
    ) -> Result<Self> {
        use std::os::unix::process::CommandExt;
        let port = validate_config(config, origin, model_name)?;
        let device_csv = query_device_identities().await?;
        let visible_devices =
            bind_device_identities(&plan.nvidia_indices, &config.gpu_uuids, &device_csv)?;
        let nonce = uuid::Uuid::new_v4().to_string();
        let mut engine_env = BTreeMap::new();
        for (key, value) in &plan.env {
            if !matches!(
                key.as_str(),
                "CUDA_VISIBLE_DEVICES"
                    | "STRATA_INDEXER_TILE"
                    | "STRATA_SERVE_DSPARK"
                    | "STRATA_DSPARK_DEPTH"
                    | "STRATA_DSPARK_CONF"
                    | "STRATA_SERVE_DSPARK_EVICT"
            ) || engine_env.insert(key.as_str(), value.as_str()).is_some()
            {
                return Err(refused(
                    "managed Xolotl environment exceeds its typed launch authority",
                ));
            }
        }
        engine_env.insert("CUDA_VISIBLE_DEVICES", visible_devices.as_str());
        let launch = Launch {
            schema: "a3s.power.xolotl.launch.v1",
            nonce: &nonce,
            parent_pid: std::process::id(),
            server: &config.server,
            tokenizer: &config.tokenizer,
            model_name,
            engine_argv: &plan.argv,
            engine_env,
            host: "127.0.0.1",
            port,
        };
        let body = serde_json::to_vec(&launch)
            .map_err(|_| refused("managed Xolotl launch cannot be encoded"))?;
        if body.len() > MAX_LAUNCH_BYTES {
            return Err(refused("managed Xolotl launch exceeds its IPC byte bound"));
        }
        let mut command = Command::new(&config.python);
        command
            .arg("-P")
            .arg("-s")
            .arg(&config.bridge)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .env("STRATA_TEXT_LOG", "0")
            .env_remove("STRATA_REQUEST_LINES");
        command.as_std_mut().process_group(0);
        let child = command
            .spawn()
            .map_err(|_| refused("managed Xolotl adapter could not start"))?;
        let group = i32::try_from(
            child
                .id()
                .ok_or_else(|| refused("managed Xolotl adapter omitted its PID"))?,
        )
        .map_err(|_| refused("managed Xolotl adapter PID overflowed"))?;
        if group <= 1 {
            return Err(refused("managed Xolotl adapter PID is unsafe"));
        }
        let mut engine = Self {
            child,
            group,
            engine_pid: 0,
            model_name: model_name.into(),
            origin: origin.trim_end_matches('/').into(),
            shutdown: Duration::from_secs(config.shutdown_timeout_seconds),
        };
        let ready =
            tokio::time::timeout(Duration::from_secs(config.startup_timeout_seconds), async {
                let mut input = engine
                    .child
                    .stdin
                    .take()
                    .ok_or_else(|| refused("managed Xolotl adapter omitted its IPC input"))?;
                input
                    .write_all(&body)
                    .await
                    .map_err(|_| refused("managed Xolotl launch IPC failed"))?;
                input
                    .shutdown()
                    .await
                    .map_err(|_| refused("managed Xolotl launch IPC did not close"))?;
                drop(input);
                let output = engine.child.stdout.take().ok_or_else(|| {
                    refused("managed Xolotl adapter omitted its readiness channel")
                })?;
                let mut bytes = Vec::new();
                output
                    .take(MAX_READY_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .await
                    .map_err(|_| refused("managed Xolotl readiness IPC failed"))?;
                if bytes.len() as u64 > MAX_READY_BYTES {
                    return Err(refused("managed Xolotl readiness exceeds its byte bound"));
                }
                let proof: Ready = serde_json::from_slice(&bytes).map_err(|_| {
                    refused("managed Xolotl adapter omitted its exact readiness proof")
                })?;
                if proof.schema != "a3s.power.xolotl.ready.v2"
                    || proof.nonce != nonce
                    || proof.model_name != model_name
                    || proof.engine_pid <= 1
                {
                    return Err(refused(
                        "managed Xolotl readiness changed its process/model identity",
                    ));
                }
                validate_context_proof(plan, proof.context_tokens)?;
                let expected_dspark = plan.env.iter().any(|(key, _)| key == "STRATA_SERVE_DSPARK");
                if proof.dspark_active != expected_dspark {
                    return Err(refused(
                        "managed Xolotl did not prove its admitted DSpark state",
                    ));
                }
                engine.engine_pid = i32::try_from(proof.engine_pid)
                    .map_err(|_| refused("managed Xolotl engine PID overflowed"))?;
                loop {
                    engine.ensure_alive()?;
                    if engine.probe(http).await.is_ok() {
                        return Ok(());
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            })
            .await
            .map_err(|_| {
                refused("managed Xolotl engine did not become ready before its startup deadline")
            })?;
        ready?;
        Ok(engine)
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) async fn start(
        _: &XolotlManagedConfig,
        _: &XolotlLaunchPlan,
        _: &str,
        _: &str,
        _: &reqwest::Client,
    ) -> Result<Self> {
        Err(refused(
            "managed Xolotl currently requires Linux process supervision",
        ))
    }

    pub(super) fn ensure_alive(&mut self) -> Result<()> {
        if self
            .child
            .try_wait()
            .map_err(|_| refused("managed Xolotl process status could not be observed"))?
            .is_some()
        {
            return Err(refused("managed Xolotl adapter has exited"));
        }
        #[cfg(target_os = "linux")]
        {
            // The adapter's child must remain in this exact unreaped owner group.
            if self.engine_pid <= 1 || unsafe { libc::getpgid(self.engine_pid) } != self.group {
                return Err(refused(
                    "managed Xolotl resident engine is no longer owned or alive",
                ));
            }
        }
        Ok(())
    }
    pub(super) async fn probe(&mut self, http: &reqwest::Client) -> Result<()> {
        self.ensure_alive()?;
        let mut response = http
            .get(format!("{}/health", self.origin))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|_| refused("managed Xolotl readiness endpoint is unavailable"))?;
        if !response.status().is_success() {
            return Err(refused(
                "managed Xolotl readiness endpoint refused the probe",
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| refused("managed Xolotl readiness body failed"))?
        {
            if bytes.len().saturating_add(chunk.len()) > MAX_HEALTH_BYTES {
                return Err(refused(
                    "managed Xolotl readiness body exceeds its byte bound",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let health: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| refused("managed Xolotl readiness body is invalid"))?;
        if health.get("loaded").and_then(|v| v.as_bool()) != Some(true)
            || health.get("model").and_then(|v| v.as_str()) != Some(self.model_name.as_str())
            || health.get("service").and_then(|v| v.as_str()) != Some("strata")
        {
            return Err(refused(
                "managed Xolotl readiness did not prove this model is loaded",
            ));
        }
        self.ensure_alive()
    }
    pub(super) async fn stop(&mut self) -> Result<()> {
        if self
            .child
            .try_wait()
            .map_err(|_| refused("managed Xolotl shutdown could not observe its owner"))?
            .is_some()
        {
            return Ok(());
        }
        self.signal_group(libc::SIGTERM);
        match tokio::time::timeout(self.shutdown, self.child.wait()).await {
            Ok(result) => {
                result.map_err(|_| refused("managed Xolotl shutdown could not reap its owner"))?;
                Ok(())
            }
            Err(_) => {
                // The owner is not reaped, so its numeric group cannot be reused.
                self.signal_group(libc::SIGKILL);
                tokio::time::timeout(Duration::from_secs(2), self.child.wait())
                    .await
                    .map_err(|_| refused("managed Xolotl shutdown exceeded its kill deadline"))?
                    .map_err(|_| refused("managed Xolotl killed owner could not be reaped"))?;
                Ok(())
            }
        }
    }
    fn signal_group(&self, signal: i32) {
        #[cfg(target_os = "linux")]
        if self.group > 1 {
            unsafe {
                libc::kill(-self.group, signal);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = signal;
    }
}
impl Drop for ManagedEngine {
    fn drop(&mut self) {
        // Never signal a numeric process group after the owner has been reaped.
        if matches!(self.child.try_wait(), Ok(None)) {
            self.signal_group(libc::SIGKILL);
        }
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
#[path = "managed_tests.rs"]
mod tests;
