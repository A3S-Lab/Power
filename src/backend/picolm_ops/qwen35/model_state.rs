use super::attention::{Qwen35AttentionConfig, Qwen35AttentionState};
use super::state::{Qwen35RecurrentConfig, Qwen35RecurrentState};
use crate::backend::gguf_stream::{GgufMeta, Qwen35LayerKind};
use crate::error::{PowerError, Result};

/// Architecture-owned mutable state for one Qwen3.5 layer.
#[derive(Debug, Clone)]
pub enum Qwen35LayerState {
    Recurrent(Qwen35RecurrentState),
    FullAttention(Qwen35AttentionState),
    Mtp(Qwen35AttentionState),
}

impl Qwen35LayerState {
    pub fn clear(&mut self) {
        match self {
            Self::Recurrent(state) => state.clear(),
            Self::FullAttention(state) | Self::Mtp(state) => state.clear(),
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Recurrent(state) => state.memory_bytes(),
            Self::FullAttention(state) | Self::Mtp(state) => state.memory_bytes(),
        }
    }
}

/// Per-sequence state whose layer variants exactly follow typed GGUF metadata.
#[derive(Debug, Clone)]
pub struct Qwen35ModelState {
    layers: Vec<Qwen35LayerState>,
}

impl Qwen35ModelState {
    pub fn from_metadata(meta: &GgufMeta, max_sequence_length: usize) -> Result<Self> {
        let architecture = meta.architecture.qwen35().ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "qwen35 model state requires architecture 'qwen35', got '{}'",
                meta.architecture.name()
            ))
        })?;
        if architecture.total_layer_count() != meta.n_layers as usize {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 model state layer plan has {} entries, expected {}",
                architecture.total_layer_count(),
                meta.n_layers
            )));
        }

        let recurrent_config = Qwen35RecurrentConfig::from_architecture(architecture)?;
        let attention_config = Qwen35AttentionConfig::from_metadata(meta, max_sequence_length)?;
        let mut layers = Vec::with_capacity(architecture.total_layer_count());
        for kind in architecture.layer_kinds() {
            let layer = match kind {
                Qwen35LayerKind::Recurrent => {
                    Qwen35LayerState::Recurrent(Qwen35RecurrentState::new(recurrent_config)?)
                }
                Qwen35LayerKind::FullAttention => {
                    Qwen35LayerState::FullAttention(Qwen35AttentionState::new(attention_config))
                }
                Qwen35LayerKind::Mtp => {
                    Qwen35LayerState::Mtp(Qwen35AttentionState::new(attention_config))
                }
            };
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layer(&self, index: usize) -> Option<&Qwen35LayerState> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Qwen35LayerState> {
        self.layers.get_mut(index)
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(Qwen35LayerState::clear);
    }

    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().fold(0usize, |total, layer| {
            total.saturating_add(layer.memory_bytes())
        })
    }
}

impl Drop for Qwen35ModelState {
    fn drop(&mut self) {
        self.clear();
    }
}
