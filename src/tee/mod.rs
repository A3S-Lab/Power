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
pub mod key_provider;
pub mod model_seal;
pub mod policy;
pub mod privacy;
