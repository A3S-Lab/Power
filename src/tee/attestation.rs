//! TEE attestation: detect hardware TEE and generate attestation reports.

use serde::{Deserialize, Serialize};

/// Type of TEE hardware detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeeType {
    /// AMD Secure Encrypted Virtualization - Secure Nested Paging.
    SevSnp,
    /// Intel Trust Domain Extensions.
    Tdx,
    /// Simulated TEE for development (A3S_TEE_SIMULATE=1).
    Simulated,
    /// No TEE detected.
    None,
}

impl std::fmt::Display for TeeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeeType::SevSnp => write!(f, "sev-snp"),
            TeeType::Tdx => write!(f, "tdx"),
            TeeType::Simulated => write!(f, "simulated"),
            TeeType::None => write!(f, "none"),
        }
    }
}

/// Remote attestation report from the TEE hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    /// Report format version (current: "1.0").
    #[serde(default = "default_report_version")]
    pub version: String,
    pub tee_type: TeeType,
    /// Raw report data from the TEE hardware (includes nonce when provided).
    #[serde(with = "hex_bytes")]
    pub report_data: Vec<u8>,
    /// Platform measurement (launch digest).
    #[serde(with = "hex_bytes")]
    pub measurement: Vec<u8>,
    /// Full raw attestation report from firmware (for client-side verification).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "hex_bytes_opt"
    )]
    pub raw_report: Option<Vec<u8>>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Client-supplied nonce bound into this report (prevents replay attacks).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "hex_bytes_opt"
    )]
    pub nonce: Option<Vec<u8>>,
}

fn default_report_version() -> String {
    "1.0".to_string()
}

/// Trait for TEE attestation providers.
#[async_trait::async_trait]
pub trait TeeProvider: Send + Sync {
    /// Generate a remote attestation report.
    ///
    /// If `nonce` is provided it is bound into `report_data` so the report
    /// is unique to this request and cannot be replayed.
    ///
    /// If `model_hash` is provided (SHA-256 of the model file), it is bound
    /// into `report_data` alongside the nonce, tying the attestation to the
    /// specific model being served.
    async fn attestation_report(
        &self,
        nonce: Option<&[u8]>,
    ) -> crate::error::Result<AttestationReport>;

    /// Generate an attestation report bound to a specific model hash.
    ///
    /// Calls `attestation_report` with a combined `report_data` that includes
    /// both the nonce and the model's SHA-256 hash.
    async fn attestation_report_with_model(
        &self,
        nonce: Option<&[u8]>,
        model_hash: Option<&[u8]>,
    ) -> crate::error::Result<AttestationReport> {
        // Build combined report_data: [nonce(32)][model_hash(32)]
        let combined = build_report_data(nonce, model_hash);
        self.attestation_report(Some(&combined)).await
    }

    /// Whether we are running inside a TEE.
    fn is_tee_environment(&self) -> bool;
    /// The type of TEE hardware detected.
    fn tee_type(&self) -> TeeType;
}

/// Default TEE provider that auto-detects hardware from device files
/// or the `A3S_TEE_SIMULATE` environment variable.
pub struct DefaultTeeProvider {
    pub(crate) detected: TeeType,
}

impl DefaultTeeProvider {
    pub fn detect() -> Self {
        let detected = detect_tee_type();
        if detected != TeeType::None {
            tracing::info!(tee_type = %detected, "TEE environment detected");
        }
        Self { detected }
    }

    /// Create a provider with a specific TEE type (for testing).
    pub fn with_type(tee_type: TeeType) -> Self {
        Self { detected: tee_type }
    }
}

#[async_trait::async_trait]
impl TeeProvider for DefaultTeeProvider {
    async fn attestation_report(
        &self,
        nonce: Option<&[u8]>,
    ) -> crate::error::Result<AttestationReport> {
        match self.detected {
            TeeType::SevSnp => read_sev_snp_report(nonce).await,
            TeeType::Tdx => read_tdx_report(nonce).await,
            TeeType::Simulated => Ok(simulated_report(nonce)),
            TeeType::None => Err(crate::error::PowerError::Config(
                "No TEE environment detected; cannot generate attestation report".to_string(),
            )),
        }
    }

    fn is_tee_environment(&self) -> bool {
        self.detected != TeeType::None
    }

    fn tee_type(&self) -> TeeType {
        self.detected
    }
}

/// Detect TEE type from hardware device files or environment.
fn detect_tee_type() -> TeeType {
    // Check for simulation mode first (development)
    if std::env::var("A3S_TEE_SIMULATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return TeeType::Simulated;
    }

    // AMD SEV-SNP: /dev/sev-guest
    if std::path::Path::new("/dev/sev-guest").exists() {
        return TeeType::SevSnp;
    }

    // Intel TDX: /dev/tdx_guest or /dev/tdx-guest
    if std::path::Path::new("/dev/tdx_guest").exists()
        || std::path::Path::new("/dev/tdx-guest").exists()
    {
        return TeeType::Tdx;
    }

    TeeType::None
}

/// Embed an optional nonce into a 64-byte report_data buffer.
///
/// Nonce bytes are copied into the first N bytes of the buffer
/// (zero-padded to 64). This binds the report to the caller's nonce.
fn embed_nonce(nonce: Option<&[u8]>, base: u8) -> Vec<u8> {
    let mut data = vec![base; 64];
    if let Some(n) = nonce {
        let len = n.len().min(64);
        data[..len].copy_from_slice(&n[..len]);
    }
    data
}

/// Build a 64-byte `report_data` buffer binding an optional nonce and model hash.
///
/// Layout: `[nonce (32 bytes, zero-padded)][model_sha256 (32 bytes, zero-padded)]`
///
/// This binds the attestation report to both the client's replay-prevention nonce
/// and the specific model being served, preventing cross-model report reuse.
pub fn build_report_data(nonce: Option<&[u8]>, model_hash: Option<&[u8]>) -> Vec<u8> {
    let mut data = vec![0u8; 64];
    if let Some(n) = nonce {
        let len = n.len().min(32);
        data[..len].copy_from_slice(&n[..len]);
    }
    if let Some(h) = model_hash {
        let len = h.len().min(32);
        data[32..32 + len].copy_from_slice(&h[..len]);
    }
    data
}

/// Read attestation report from AMD SEV-SNP guest device.
///
/// On Linux, opens `/dev/sev-guest` and issues the `SNP_GET_REPORT` ioctl.
/// On non-Linux platforms, returns an error (SEV-SNP is Linux-only).
async fn read_sev_snp_report(nonce: Option<&[u8]>) -> crate::error::Result<AttestationReport> {
    #[cfg(target_os = "linux")]
    {
        let nonce_owned = nonce.map(|n| n.to_vec());
        // ioctl is blocking — run on a blocking thread
        let (report_data, measurement, raw_report) = tokio::task::spawn_blocking(move || {
            sev_snp_ioctl::snp_get_report(nonce_owned.as_deref())
        })
        .await
        .map_err(|e| {
            crate::error::PowerError::Config(format!("SEV-SNP attestation task failed: {e}"))
        })??;

        tracing::info!("SEV-SNP attestation report generated successfully");
        Ok(AttestationReport {
            version: default_report_version(),
            tee_type: TeeType::SevSnp,
            report_data,
            measurement,
            raw_report: Some(raw_report),
            timestamp: chrono::Utc::now(),
            nonce: nonce.map(|n| n.to_vec()),
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = nonce;
        Err(crate::error::PowerError::Config(
            "SEV-SNP attestation is only available on Linux".to_string(),
        ))
    }
}

/// Read attestation report from Intel TDX guest device.
///
/// On Linux, opens `/dev/tdx-guest` (or `/dev/tdx_guest`) and issues
/// the `TDX_CMD_GET_REPORT0` ioctl.
/// On non-Linux platforms, returns an error (TDX is Linux-only).
async fn read_tdx_report(nonce: Option<&[u8]>) -> crate::error::Result<AttestationReport> {
    #[cfg(target_os = "linux")]
    {
        let nonce_owned = nonce.map(|n| n.to_vec());
        let (report_data, measurement, raw_report) =
            tokio::task::spawn_blocking(move || tdx_ioctl::tdx_get_report(nonce_owned.as_deref()))
                .await
                .map_err(|e| {
                    crate::error::PowerError::Config(format!("TDX attestation task failed: {e}"))
                })??;

        tracing::info!("TDX attestation report generated successfully");
        Ok(AttestationReport {
            version: default_report_version(),
            tee_type: TeeType::Tdx,
            report_data,
            measurement,
            raw_report: Some(raw_report),
            timestamp: chrono::Utc::now(),
            nonce: nonce.map(|n| n.to_vec()),
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = nonce;
        Err(crate::error::PowerError::Config(
            "TDX attestation is only available on Linux".to_string(),
        ))
    }
}

/// Generate a simulated attestation report for development.
fn simulated_report(nonce: Option<&[u8]>) -> AttestationReport {
    AttestationReport {
        version: default_report_version(),
        tee_type: TeeType::Simulated,
        report_data: embed_nonce(nonce, 0xAA),
        measurement: vec![0xBB; 48],
        raw_report: None,
        timestamp: chrono::Utc::now(),
        nonce: nonce.map(|n| n.to_vec()),
    }
}

/// Linux SEV-SNP guest ioctl interface.
///
/// Wraps the kernel ABI from `include/uapi/linux/sev-guest.h`:
/// - `SNP_GET_REPORT` ioctl on `/dev/sev-guest`
/// - Parses `report_data` (offset 0x50) and `measurement` (offset 0x90) from the response
#[cfg(target_os = "linux")]
mod sev_snp_ioctl {
    use std::os::unix::io::AsRawFd;

    // Kernel struct: snp_report_req (96 bytes)
    #[repr(C)]
    pub struct SnpReportReq {
        pub user_data: [u8; 64], // arbitrary data bound into the report
        pub vmpl: u32,           // VMPL level (0 = most privileged)
        pub rsvd: [u8; 28],      // reserved, must be zero
    }

    // Kernel struct: snp_report_resp (4000 bytes)
    #[repr(C)]
    pub struct SnpReportResp {
        pub data: [u8; 4000], // raw attestation report from firmware
    }

    // Kernel struct: snp_guest_request_ioctl
    // Note: the union { exitinfo2; { fw_error; vmm_error } } is represented
    // as two u32 fields since we only need to read fw_error on failure.
    #[repr(C)]
    pub struct SnpGuestRequestIoctl {
        pub msg_version: u8, // must be 1
        pub _pad: [u8; 7],   // padding to align req_data to 8 bytes
        pub req_data: u64,   // pointer to SnpReportReq
        pub resp_data: u64,  // pointer to SnpReportResp
        pub fw_error: u32,   // firmware error code (lower 32 bits of exitinfo2)
        pub vmm_error: u32,  // VMM error code (upper 32 bits of exitinfo2)
    }

    // SNP_GET_REPORT = _IOWR('S', 0x0, struct snp_guest_request_ioctl)
    // _IOWR(type, nr, size) = (3 << 30) | (size << 16) | (type << 8) | nr
    // size = size_of::<SnpGuestRequestIoctl>() = 32 bytes
    nix::ioctl_readwrite!(snp_get_report_ioctl, b'S', 0x0, SnpGuestRequestIoctl);

    // Attestation report field offsets (AMD SEV-SNP Firmware ABI Spec, Table 23)
    const REPORT_DATA_OFFSET: usize = 0x50; // 64 bytes: echoed user_data / nonce
    const MEASUREMENT_OFFSET: usize = 0x90; // 48 bytes: launch digest

    /// Issue SNP_GET_REPORT ioctl and return (report_data, measurement, raw_report).
    pub fn snp_get_report(
        nonce: Option<&[u8]>,
    ) -> crate::error::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // Open /dev/sev-guest
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/sev-guest")
            .map_err(|e| {
                crate::error::PowerError::Config(format!("Failed to open /dev/sev-guest: {e}"))
            })?;

        let mut req = SnpReportReq {
            user_data: [0u8; 64],
            vmpl: 0,
            rsvd: [0u8; 28],
        };

        // Embed nonce into user_data
        if let Some(n) = nonce {
            let len = n.len().min(64);
            req.user_data[..len].copy_from_slice(&n[..len]);
        }

        let mut resp = SnpReportResp { data: [0u8; 4000] };

        let mut guest_req = SnpGuestRequestIoctl {
            msg_version: 1,
            _pad: [0u8; 7],
            req_data: &req as *const SnpReportReq as u64,
            resp_data: &mut resp as *mut SnpReportResp as u64,
            fw_error: 0,
            vmm_error: 0,
        };

        // Safety: we pass valid pointers to correctly-sized structs.
        // The ioctl writes into resp.data which we own on the stack.
        let ret = unsafe { snp_get_report_ioctl(file.as_raw_fd(), &mut guest_req) };

        match ret {
            Ok(_) => {}
            Err(e) => {
                return Err(crate::error::PowerError::Config(format!(
                    "SNP_GET_REPORT ioctl failed: {e} (fw_error=0x{:08x}, vmm_error=0x{:08x})",
                    guest_req.fw_error, guest_req.vmm_error
                )));
            }
        }

        // Validate the response buffer is large enough to hold the fields we need
        let min_size = MEASUREMENT_OFFSET + 48;
        if resp.data.len() < min_size {
            return Err(crate::error::PowerError::Config(format!(
                "SEV-SNP response too small: {} bytes (need at least {min_size})",
                resp.data.len()
            )));
        }

        let report_data = resp.data[REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + 64].to_vec();
        let measurement = resp.data[MEASUREMENT_OFFSET..MEASUREMENT_OFFSET + 48].to_vec();

        // Determine actual report size: the firmware fills from offset 0 up to
        // the signature. We return the full 4000-byte buffer trimmed to the
        // standard report size (1184 bytes per AMD spec) if it fits, otherwise
        // the full buffer.
        const AMD_REPORT_SIZE: usize = 1184;
        let raw_report = if resp.data.len() >= AMD_REPORT_SIZE {
            resp.data[..AMD_REPORT_SIZE].to_vec()
        } else {
            resp.data.to_vec()
        };

        Ok((report_data, measurement, raw_report))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_snp_report_req_size() {
            // user_data(64) + vmpl(4) + rsvd(28) = 96
            assert_eq!(size_of::<SnpReportReq>(), 96);
        }

        #[test]
        fn test_snp_report_resp_size() {
            assert_eq!(size_of::<SnpReportResp>(), 4000);
        }

        #[test]
        fn test_snp_guest_request_ioctl_size() {
            // msg_version(1) + pad(7) + req_data(8) + resp_data(8) + fw_error(4) + vmm_error(4) = 32
            assert_eq!(size_of::<SnpGuestRequestIoctl>(), 32);
        }

        #[test]
        fn test_report_field_offsets_within_resp() {
            // Verify the offsets we use are within the 4000-byte response buffer
            assert!(REPORT_DATA_OFFSET + 64 <= 4000);
            assert!(MEASUREMENT_OFFSET + 48 <= 4000);
        }

        #[test]
        fn test_snp_get_report_no_device() {
            // On a non-SEV machine, /dev/sev-guest doesn't exist → Config error
            if !std::path::Path::new("/dev/sev-guest").exists() {
                let result = snp_get_report(None);
                assert!(result.is_err());
                let msg = result.unwrap_err().to_string();
                assert!(msg.contains("/dev/sev-guest"), "error: {msg}");
            }
        }
    }
}

/// Intel TDX guest ioctl interface.
///
/// Wraps the kernel ABI from `include/uapi/linux/tdx-guest.h`:
/// - `TDX_CMD_GET_REPORT0` ioctl on `/dev/tdx-guest`
/// - Parses `reportdata` (offset 64) and `mrtd` (offset 528) from the TDREPORT
#[cfg(target_os = "linux")]
mod tdx_ioctl {
    use std::os::unix::io::AsRawFd;

    // Kernel struct: tdx_report_req (1088 bytes)
    // TDX_REPORTDATA_LEN = 64, TDX_REPORT_LEN = 1024
    #[repr(C)]
    pub struct TdxReportReq {
        pub reportdata: [u8; 64], // user-supplied data bound into the report
        pub tdreport: [u8; 1024], // output: TDREPORT from TDX module
    }

    // TDX_CMD_GET_REPORT0 = _IOWR('T', 1, struct tdx_report_req)
    nix::ioctl_readwrite!(tdx_cmd_get_report0, b'T', 1, TdxReportReq);

    // TDREPORT field offsets (Intel TDX Module Specification)
    //
    // TDREPORT_STRUCT layout (1024 bytes):
    //   [0..256]   REPORTMACSTRUCT (256 bytes)
    //     [64..128]  reportdata (64 bytes) — echoed user input
    //   [256..495] TEE_TCB_INFO (239 bytes)
    //   [495..512] reserved (17 bytes)
    //   [512..1024] TDINFO_STRUCT (512 bytes)
    //     [512..520]  attr (8 bytes)
    //     [520..528]  xfam (8 bytes)
    //     [528..576]  mrtd (48 bytes) — TD measurement
    const REPORTDATA_OFFSET: usize = 64; // within REPORTMACSTRUCT
    const MRTD_OFFSET: usize = 528; // within TDINFO_STRUCT (512 + 8 + 8)

    /// Issue TDX_CMD_GET_REPORT0 ioctl and return (report_data, measurement, raw_report).
    pub fn tdx_get_report(
        nonce: Option<&[u8]>,
    ) -> crate::error::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // Try /dev/tdx-guest first (newer kernels), fall back to /dev/tdx_guest
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/tdx-guest")
            .or_else(|_| {
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open("/dev/tdx_guest")
            })
            .map_err(|e| {
                crate::error::PowerError::Config(format!(
                    "Failed to open /dev/tdx-guest or /dev/tdx_guest: {e}"
                ))
            })?;

        let mut req = TdxReportReq {
            reportdata: [0u8; 64],
            tdreport: [0u8; 1024],
        };

        // Embed nonce into reportdata
        if let Some(n) = nonce {
            let len = n.len().min(64);
            req.reportdata[..len].copy_from_slice(&n[..len]);
        }

        // Safety: we pass a valid pointer to a correctly-sized struct.
        // The ioctl writes into req.tdreport which we own on the stack.
        let ret = unsafe { tdx_cmd_get_report0(file.as_raw_fd(), &mut req) };

        match ret {
            Ok(_) => {}
            Err(e) => {
                return Err(crate::error::PowerError::Config(format!(
                    "TDX_CMD_GET_REPORT0 ioctl failed: {e}"
                )));
            }
        }

        let report_data = req.tdreport[REPORTDATA_OFFSET..REPORTDATA_OFFSET + 64].to_vec();
        let measurement = req.tdreport[MRTD_OFFSET..MRTD_OFFSET + 48].to_vec();
        let raw_report = req.tdreport.to_vec();

        Ok((report_data, measurement, raw_report))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_tdx_report_req_size() {
            // reportdata(64) + tdreport(1024) = 1088
            assert_eq!(size_of::<TdxReportReq>(), 1088);
        }

        #[test]
        fn test_tdreport_field_offsets_within_bounds() {
            assert!(REPORTDATA_OFFSET + 64 <= 1024);
            assert!(MRTD_OFFSET + 48 <= 1024);
        }

        #[test]
        fn test_tdx_get_report_no_device() {
            // On a non-TDX machine, neither device exists → Config error
            let has_tdx = std::path::Path::new("/dev/tdx-guest").exists()
                || std::path::Path::new("/dev/tdx_guest").exists();
            if !has_tdx {
                let result = tdx_get_report(None);
                assert!(result.is_err());
                let msg = result.unwrap_err().to_string();
                assert!(msg.contains("tdx"), "error: {msg}");
            }
        }
    }
}

/// Hex serialization for byte vectors in attestation reports.
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        s.serialize_str(&hex)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let hex = String::deserialize(d)?;
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(serde::de::Error::custom))
            .collect()
    }
}

/// Hex serialization for optional byte vectors in attestation reports.
mod hex_bytes_opt {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match bytes {
            Some(b) => {
                let hex: String = b.iter().map(|byte| format!("{byte:02x}")).collect();
                s.serialize_str(&hex)
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let hex = Option::<String>::deserialize(d)?;
        match hex {
            None => Ok(None),
            Some(h) => (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).map_err(serde::de::Error::custom))
                .collect::<Result<Vec<u8>, _>>()
                .map(Some),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_type_display() {
        assert_eq!(TeeType::SevSnp.to_string(), "sev-snp");
        assert_eq!(TeeType::Tdx.to_string(), "tdx");
        assert_eq!(TeeType::Simulated.to_string(), "simulated");
        assert_eq!(TeeType::None.to_string(), "none");
    }

    #[test]
    fn test_tee_type_serde_roundtrip() {
        let json = serde_json::to_string(&TeeType::SevSnp).unwrap();
        assert_eq!(json, "\"sev-snp\"");
        let parsed: TeeType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, TeeType::SevSnp);
    }

    #[test]
    fn test_simulated_report() {
        let report = simulated_report(None);
        assert_eq!(report.tee_type, TeeType::Simulated);
        assert_eq!(report.report_data.len(), 64);
        assert_eq!(report.measurement.len(), 48);
        assert!(report.nonce.is_none());
    }

    #[test]
    fn test_simulated_report_with_nonce() {
        let nonce = b"test-nonce-12345";
        let report = simulated_report(Some(nonce));
        assert_eq!(report.report_data[..nonce.len()], *nonce);
        // Remaining bytes are still 0xAA baseline
        assert!(report.report_data[nonce.len()..].iter().all(|&b| b == 0xAA));
        assert_eq!(report.nonce.as_deref(), Some(nonce.as_ref()));
    }

    #[test]
    fn test_embed_nonce_without_nonce() {
        let data = embed_nonce(None, 0xAA);
        assert_eq!(data, vec![0xAA; 64]);
    }

    #[test]
    fn test_embed_nonce_truncates_at_64() {
        let nonce = vec![0xFF; 100];
        let data = embed_nonce(Some(&nonce), 0x00);
        assert_eq!(data.len(), 64);
        assert!(data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_attestation_report_serialization() {
        let report = simulated_report(None);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"tee_type\":\"simulated\""));
        assert!(json.contains("\"report_data\":"));
        // nonce field omitted when None
        assert!(!json.contains("\"nonce\""));
        let parsed: AttestationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tee_type, TeeType::Simulated);
        assert_eq!(parsed.report_data, report.report_data);
    }

    #[test]
    fn test_attestation_report_serialization_with_nonce() {
        let nonce = vec![0x01, 0x02, 0x03];
        let report = simulated_report(Some(&nonce));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"nonce\":\"010203\""));
        let parsed: AttestationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.nonce, Some(nonce));
    }

    #[test]
    fn test_attestation_report_version_field() {
        let report = simulated_report(None);
        assert_eq!(report.version, "1.0");
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"version\":\"1.0\""));
    }

    #[test]
    fn test_attestation_report_version_default_on_deserialize() {
        // When version is missing in JSON (old format), it defaults to "1.0"
        let json = r#"{"tee_type":"simulated","report_data":"aabb","measurement":"ccdd","timestamp":"2024-01-01T00:00:00Z"}"#;
        let report: AttestationReport = serde_json::from_str(json).unwrap();
        assert_eq!(report.version, "1.0");
    }

    #[test]
    fn test_detect_tee_type_none_by_default() {
        // In a normal dev environment, no TEE hardware is present
        std::env::remove_var("A3S_TEE_SIMULATE");
        let tee = detect_tee_type();
        // Could be None or Simulated depending on env; just verify it doesn't panic
        assert!(tee == TeeType::None || tee == TeeType::Simulated);
    }

    #[test]
    fn test_default_provider_no_tee() {
        std::env::remove_var("A3S_TEE_SIMULATE");
        let provider = DefaultTeeProvider {
            detected: TeeType::None,
        };
        assert!(!provider.is_tee_environment());
        assert_eq!(provider.tee_type(), TeeType::None);
    }

    #[test]
    fn test_default_provider_simulated() {
        let provider = DefaultTeeProvider {
            detected: TeeType::Simulated,
        };
        assert!(provider.is_tee_environment());
        assert_eq!(provider.tee_type(), TeeType::Simulated);
    }

    #[tokio::test]
    async fn test_simulated_attestation_report() {
        let provider = DefaultTeeProvider {
            detected: TeeType::Simulated,
        };
        let report = provider.attestation_report(None).await.unwrap();
        assert_eq!(report.tee_type, TeeType::Simulated);
        assert!(report.nonce.is_none());
    }

    #[tokio::test]
    async fn test_simulated_attestation_report_with_nonce() {
        let provider = DefaultTeeProvider {
            detected: TeeType::Simulated,
        };
        let nonce = b"replay-prevention";
        let report = provider.attestation_report(Some(nonce)).await.unwrap();
        assert_eq!(report.tee_type, TeeType::Simulated);
        assert_eq!(report.nonce.as_deref(), Some(nonce.as_ref()));
        assert_eq!(report.report_data[..nonce.len()], *nonce);
    }

    #[tokio::test]
    async fn test_no_tee_attestation_fails() {
        let provider = DefaultTeeProvider {
            detected: TeeType::None,
        };
        let result = provider.attestation_report(None).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_bytes_roundtrip() {
        let report = simulated_report(None);
        let json = serde_json::to_string(&report).unwrap();
        let parsed: AttestationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.report_data, report.report_data);
        assert_eq!(parsed.measurement, report.measurement);
    }
}
