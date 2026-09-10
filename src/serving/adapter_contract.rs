//! Closed software ownership contract for injected P/D adapters.
//!
//! Types only: Empty placeholders and trait methods live with the serving ports.
//! Declaring this contract does **not** claim high-speed-network evidence or
//! production readiness.

use crate::error::{PowerError, Result};

/// Who owns registered transfer / phase memory under a production injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterMemoryOwnership {
    /// Adapter registers and frees device or host memory. Power holds only
    /// opaque local handles and content-free declared-byte accounting
    /// (`registered_adapter_bytes`); it never copies or retains KV payloads.
    AdapterOwnedRegistration,
}

/// Who owns transport integrity for opaque state movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterTransportIntegrity {
    /// Adapter owns authentication, encryption, RDMA/HSN drivers, and ticket
    /// bytes. Power never inspects KV and does not treat transport completion
    /// as decode success.
    AdapterOwned,
}

/// Cleanup confirmation required of a production injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterCleanupObligation {
    /// Abort, timeout, and compensating cleanup must reclaim adapter-owned
    /// registration. Unconfirmed cleanup taints the process-local transfer
    /// wrapper unavailable for the remainder of the generation.
    ConfirmedReclaim,
}

/// Closed software obligations a production adapter injection must satisfy.
///
/// Declaring this contract does not advertise high-speed transport or make an
/// Empty placeholder production-ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionAdapterContract {
    pub memory_ownership: AdapterMemoryOwnership,
    pub transport_integrity: AdapterTransportIntegrity,
    pub cleanup: AdapterCleanupObligation,
}

impl ProductionAdapterContract {
    /// The only closed production contract Power currently recognizes.
    pub const REQUIRED: Self = Self {
        memory_ownership: AdapterMemoryOwnership::AdapterOwnedRegistration,
        transport_integrity: AdapterTransportIntegrity::AdapterOwned,
        cleanup: AdapterCleanupObligation::ConfirmedReclaim,
    };

    pub fn validate(self) -> Result<()> {
        if self != Self::REQUIRED {
            return Err(PowerError::Config(
                "distributed serving adapters must declare the required production memory-ownership contract"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Whether a port is still an Empty placeholder or a concrete injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterProvisionState {
    /// Default software placeholder. Always reports Unavailable and refuses
    /// work. Prefill/decode composition rejects Empty until a real adapter is
    /// injected.
    Empty,
    /// Concrete adapter injection that owns the production contract surface.
    Injected,
}

impl AdapterProvisionState {
    pub fn is_empty(self) -> bool {
        matches!(self, Self::Empty)
    }

    pub fn is_injected(self) -> bool {
        matches!(self, Self::Injected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_contract_is_closed() {
        ProductionAdapterContract::REQUIRED.validate().unwrap();
    }
}
