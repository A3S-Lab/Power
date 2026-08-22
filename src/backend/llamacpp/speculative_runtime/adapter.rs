use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::speculative::{ExternalDraftSpeculative, MtpSpeculative};

use crate::error::{PowerError, Result};

pub(super) trait LlamaSpeculativeAdapter<'target, 'draft> {
    fn target_context(&self) -> &llama_cpp_2::context::LlamaContext<'target>;
    fn target_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'target>;
    fn draft_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'draft>;
    fn begin(&mut self, prompt: &[llama_cpp_2::token::LlamaToken]) -> Result<()>;
    fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<()>;
    fn draft_with_max(
        &mut self,
        n_past: i32,
        anchor: llama_cpp_2::token::LlamaToken,
        max_draft: usize,
    ) -> Result<(Vec<llama_cpp_2::token::LlamaToken>, usize)>;
    fn accept(&mut self, accepted: u16) -> Result<()>;
}

impl<'model> LlamaSpeculativeAdapter<'model, 'model> for MtpSpeculative<'model> {
    fn target_context(&self) -> &llama_cpp_2::context::LlamaContext<'model> {
        MtpSpeculative::target_context(self)
    }

    fn target_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'model> {
        MtpSpeculative::target_context_mut(self)
    }

    fn draft_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'model> {
        MtpSpeculative::draft_context_mut(self)
    }

    fn begin(&mut self, prompt: &[llama_cpp_2::token::LlamaToken]) -> Result<()> {
        MtpSpeculative::begin(self, prompt).map_err(|error| {
            PowerError::InferenceFailed(format!("Failed to begin MTP generation: {error}"))
        })
    }

    fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<()> {
        MtpSpeculative::process(self, batch).map_err(|error| {
            PowerError::InferenceFailed(format!("MTP state synchronization failed: {error}"))
        })
    }

    fn draft_with_max(
        &mut self,
        n_past: i32,
        anchor: llama_cpp_2::token::LlamaToken,
        max_draft: usize,
    ) -> Result<(Vec<llama_cpp_2::token::LlamaToken>, usize)> {
        draft_mtp_tokens(self, n_past, anchor, max_draft)
    }

    fn accept(&mut self, accepted: u16) -> Result<()> {
        MtpSpeculative::accept(self, accepted).map_err(|error| {
            PowerError::InferenceFailed(format!("Failed to commit MTP acceptance: {error}"))
        })
    }
}

impl<'target, 'draft> LlamaSpeculativeAdapter<'target, 'draft>
    for ExternalDraftSpeculative<'target, 'draft>
{
    fn target_context(&self) -> &llama_cpp_2::context::LlamaContext<'target> {
        ExternalDraftSpeculative::target_context(self)
    }

    fn target_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'target> {
        ExternalDraftSpeculative::target_context_mut(self)
    }

    fn draft_context_mut(&mut self) -> &mut llama_cpp_2::context::LlamaContext<'draft> {
        ExternalDraftSpeculative::draft_context_mut(self)
    }

    fn begin(&mut self, prompt: &[llama_cpp_2::token::LlamaToken]) -> Result<()> {
        ExternalDraftSpeculative::begin(self, prompt).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "Failed to begin external-draft generation: {error}"
            ))
        })
    }

    fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<()> {
        ExternalDraftSpeculative::process(self, batch).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "External-draft state synchronization failed: {error}"
            ))
        })
    }

    fn draft_with_max(
        &mut self,
        n_past: i32,
        anchor: llama_cpp_2::token::LlamaToken,
        max_draft: usize,
    ) -> Result<(Vec<llama_cpp_2::token::LlamaToken>, usize)> {
        ExternalDraftSpeculative::draft_with_max(self, n_past, anchor, &[], max_draft)
            .map(|drafts| (drafts, 0))
            .map_err(|error| {
                PowerError::InferenceFailed(format!("External drafting failed: {error}"))
            })
    }

    fn accept(&mut self, accepted: u16) -> Result<()> {
        ExternalDraftSpeculative::accept(self, accepted).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "Failed to commit external-draft acceptance: {error}"
            ))
        })
    }
}

fn draft_mtp_tokens(
    speculative: &mut MtpSpeculative<'_>,
    n_past: i32,
    anchor: llama_cpp_2::token::LlamaToken,
    _draft_limit: usize,
) -> Result<(Vec<llama_cpp_2::token::LlamaToken>, usize)> {
    #[cfg(feature = "llamacpp-mtp-fr")]
    let result = {
        let drafts = speculative.draft_with_max(n_past, anchor, &[], _draft_limit);
        drafts.map(|drafts| (drafts, speculative.last_recurrent_steps()))
    };

    #[cfg(not(feature = "llamacpp-mtp-fr"))]
    let result = speculative
        .draft(n_past, anchor, &[])
        .map(|drafts| (drafts, 0));

    result.map_err(|error| PowerError::InferenceFailed(format!("MTP drafting failed: {error}")))
}
