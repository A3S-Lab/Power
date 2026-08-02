mod codec;
mod source;

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

pub use codec::{
    LosslessEncodedRecord, LosslessRansNibbleHistogram, LosslessRansNibbleTable,
    LOSSLESS_RANS_FORMAT_METADATA_KEY, LOSSLESS_RANS_TABLE_METADATA_KEY,
};
#[cfg(target_os = "linux")]
pub(super) use source::physical_location;
pub use source::weight_collection_sha256;
pub(super) use source::{open_lossless_source, read_lossless_bytes};

/// Physical representation used by one verified weight source.
///
/// Lossless representations remain replicas of the mandatory canonical
/// SafeTensors primary. The digest identifies the complete stored artifact;
/// Power also verifies every decoded tensor against the primary before the
/// source becomes routable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum WeightSourceRepresentation {
    #[default]
    CanonicalSafeTensors,
    #[serde(rename = "lossless-rans-nibble-256-v1")]
    LosslessRansNibble256V1 { artifact_sha256: String },
}

impl WeightSourceRepresentation {
    pub(crate) fn validate(&self) -> Result<()> {
        if let Self::LosslessRansNibble256V1 { artifact_sha256 } = self {
            if artifact_sha256.len() != 64
                || !artifact_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(PowerError::Config(
                    "lossless weight artifact SHA-256 must contain 64 lowercase hexadecimal characters"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn artifact_sha256(&self) -> Option<&str> {
        match self {
            Self::CanonicalSafeTensors => None,
            Self::LosslessRansNibble256V1 { artifact_sha256 } => Some(artifact_sha256),
        }
    }

    pub(super) fn is_canonical(&self) -> bool {
        matches!(self, Self::CanonicalSafeTensors)
    }
}

pub(super) struct LosslessSourceState {
    pub(super) record_locations: std::collections::BTreeMap<String, super::TensorLocation>,
    pub(super) tables: Vec<LosslessRansNibbleTable>,
    pub(super) scratch_limit: u64,
}
