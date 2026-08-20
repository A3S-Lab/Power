#[cfg(feature = "llamacpp-mtp-fr")]
use super::llamacpp_context_mtp_fr_vocab;
use super::metrics::FrCoverageMetrics;
use super::{
    ensure_mtp_fr_available, llamacpp_context_output_limits, metadata_entry_enables_mtp,
    mtp_speculative_params, use_backend_greedy, use_greedy_fast_path, LlamaContextSettings,
    LlamaSamplingSettings, MtpCompletionSettings,
};
use llama_cpp_2::context::params::LlamaContextType;

fn settings() -> MtpCompletionSettings {
    MtpCompletionSettings {
        max_tokens: 16,
        stop_sequences: Vec::new(),
        draft_max: 3,
        recurrent_snapshots: 7,
        recurrent_chain: true,
        adaptive: true,
        draft_min: 0,
        draft_p_min: 0.0,
    }
}

fn sampling_settings() -> LlamaSamplingSettings {
    LlamaSamplingSettings {
        response_format: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        repeat_last_n: 64,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        temperature: 0.0,
        top_k: None,
        typical_p: None,
        top_p: 1.0,
        min_p: None,
        seed: 42,
    }
}

#[test]
fn fr_prefix_metrics_do_not_claim_ranked_vocabulary_membership() {
    let mut metrics = FrCoverageMetrics::default();
    metrics.observe_target_sample(100, Some(8192));
    metrics.observe_target_sample(9000, Some(8192));
    metrics.observe_rejection(100, Some(8192));
    metrics.observe_rejection(9000, Some(8192));
    metrics.observe_target_sample(9000, None);

    assert_eq!(metrics.target_samples, 2);
    assert_eq!(metrics.target_samples_in_token_id_prefix, 1);
    assert_eq!(metrics.rejected_rounds, 2);
    assert_eq!(metrics.corrections_outside_token_id_prefix, 1);
}

#[test]
fn greedy_fast_path_requires_unfiltered_zero_temperature_sampling() {
    assert!(use_greedy_fast_path(&sampling_settings()));

    for settings in [
        LlamaSamplingSettings {
            temperature: 0.7,
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            top_k: Some(20),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            typical_p: Some(0.95),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            top_p: 0.9,
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            min_p: Some(0.05),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            mirostat: Some(1),
            ..sampling_settings()
        },
    ] {
        assert!(!use_greedy_fast_path(&settings));
    }
}

#[test]
fn backend_greedy_requires_a_stateless_sampler_chain() {
    assert!(use_backend_greedy(&sampling_settings()));

    for settings in [
        LlamaSamplingSettings {
            response_format: Some(serde_json::json!({"type": "json_object"})),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            repeat_penalty: Some(1.1),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            frequency_penalty: Some(0.1),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            presence_penalty: Some(0.1),
            ..sampling_settings()
        },
        LlamaSamplingSettings {
            temperature: 0.7,
            ..sampling_settings()
        },
    ] {
        assert!(!use_backend_greedy(&settings));
    }
}

#[test]
fn mtp_metadata_detection_is_architecture_neutral() {
    assert!(metadata_entry_enables_mtp(
        "qwen35.nextn_predict_layers",
        "1"
    ));
    assert!(metadata_entry_enables_mtp(
        "future_arch.nextn_predict_layers",
        "3"
    ));
    assert!(!metadata_entry_enables_mtp(
        "qwen35.nextn_predict_layers",
        "0"
    ));
    assert!(!metadata_entry_enables_mtp(
        "general.architecture",
        "qwen35"
    ));
    assert!(!metadata_entry_enables_mtp(
        "future_arch.nextn_predict_layers",
        "invalid"
    ));
}

#[test]
fn mtp_parameters_accept_adapter_defaults() {
    let params = mtp_speculative_params(&settings(), true).unwrap();
    assert_eq!(params.upstream.n_max, 3);
    assert_eq!(params.upstream.n_min, 0);
    assert_eq!(params.upstream.p_min, 0.0);
    assert_eq!(params.draft_greedy, cfg!(feature = "llamacpp-mtp-fr"));
    assert_eq!(params.recurrent_draft, cfg!(feature = "llamacpp-mtp-fr"));

    let non_greedy = mtp_speculative_params(&settings(), false).unwrap();
    assert!(!non_greedy.draft_greedy);

    let thresholded = mtp_speculative_params(
        &MtpCompletionSettings {
            draft_p_min: 0.1,
            ..settings()
        },
        true,
    )
    .unwrap();
    assert!(!thresholded.draft_greedy);

    let host_staged = mtp_speculative_params(
        &MtpCompletionSettings {
            recurrent_chain: false,
            ..settings()
        },
        true,
    )
    .unwrap();
    assert_eq!(host_staged.draft_greedy, cfg!(feature = "llamacpp-mtp-fr"));
    assert!(!host_staged.recurrent_draft);
}

#[test]
fn mtp_target_context_reserves_recurrent_tail_and_exact_output_rows() {
    let params = LlamaContextSettings {
        ctx_size: 4096,
        num_batch: Some(4),
        num_thread: Some(10),
        num_thread_batch: Some(10),
        flash_attention: true,
        mtp_fr_vocab_size: None,
    }
    .params(LlamaContextType::Default, 3, 5, 4);

    assert_eq!(params.n_batch(), 5);
    assert_eq!(params.n_ctx().map(std::num::NonZeroU32::get), Some(4096));
    assert_eq!(params.n_seq_max(), 1);
    assert_eq!(params.n_rs_seq(), 3);
    assert_eq!(llamacpp_context_output_limits(&params), (4, 4));
}

#[cfg(not(feature = "llamacpp-mtp-fr"))]
#[test]
fn reduced_vocabulary_requires_the_explicit_patched_feature() {
    assert!(ensure_mtp_fr_available(None).is_ok());
    let error = ensure_mtp_fr_available(Some(8192)).unwrap_err();
    assert!(error.to_string().contains("llamacpp-mtp-fr"));
}

#[cfg(feature = "llamacpp-mtp-fr")]
#[test]
fn patched_feature_accepts_reduced_vocabulary() {
    assert!(ensure_mtp_fr_available(Some(8192)).is_ok());
}

#[cfg(feature = "llamacpp-mtp-fr")]
#[test]
fn mtp_context_applies_reduced_vocabulary_only_to_the_draft() {
    let target = LlamaContextSettings {
        ctx_size: 4096,
        num_batch: Some(14),
        num_thread: Some(10),
        num_thread_batch: Some(10),
        flash_attention: true,
        mtp_fr_vocab_size: Some(8192),
    }
    .params(LlamaContextType::Default, 6, 9, 8);
    assert_eq!(llamacpp_context_mtp_fr_vocab(&target), 0);

    let params = LlamaContextSettings {
        ctx_size: 4096,
        num_batch: Some(14),
        num_thread: Some(10),
        num_thread_batch: Some(10),
        flash_attention: true,
        mtp_fr_vocab_size: Some(8192),
    }
    .params(LlamaContextType::Mtp, 0, 9, 1);
    assert_eq!(llamacpp_context_mtp_fr_vocab(&params), 8192);
}

#[test]
fn mtp_parameters_reject_minimum_above_adapter_default() {
    let error = mtp_speculative_params(
        &MtpCompletionSettings {
            draft_min: 4,
            ..settings()
        },
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must not exceed"));
}

#[test]
fn mtp_parameters_reject_invalid_programmatic_values() {
    for draft_max in [0, 65] {
        let error = mtp_speculative_params(
            &MtpCompletionSettings {
                draft_max,
                ..settings()
            },
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("between 1 and 64"));
    }

    let error = mtp_speculative_params(
        &MtpCompletionSettings {
            draft_p_min: f32::NAN,
            ..settings()
        },
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("finite"));

    for recurrent_snapshots in [0, 65] {
        let error = mtp_speculative_params(
            &MtpCompletionSettings {
                recurrent_snapshots,
                ..settings()
            },
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("recurrent_snapshots"));
    }
}
