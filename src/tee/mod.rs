//! TEE (Trusted Execution Environment) privacy protection layer.
//!
//! Provides hardware-level privacy guarantees for LLM inference:
//! - Remote attestation (AMD SEV-SNP / Intel TDX)
//! - Model integrity verification (SHA-256)
//! - Log redaction (strip inference content from logs)
//! - Memory zeroing (zeroize sensitive buffers)
//! - Encrypted model loading (AES-256-GCM)
//! - RA-TLS transport (self-signed cert with attestation extension, `tls` feature)
//! - EPC memory detection for TEE-aware backend routing (`epc` module)

pub mod attestation;
#[cfg(feature = "tls")]
pub mod cert;
pub mod encrypted_model;
pub mod epc;
#[cfg(feature = "server")]
pub mod gpu;
#[cfg(feature = "server")]
pub mod key_provider;
#[cfg(feature = "server")]
pub mod model_seal;
pub mod model_signature;
pub mod policy;
pub mod privacy;
pub(crate) mod tdx_report;
