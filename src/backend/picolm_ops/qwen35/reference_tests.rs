use super::reference::l2_normalize;
use super::*;
use crate::backend::gguf_stream::{
    GgufArchitecture, GgufMeta, Qwen35Architecture, Qwen35ArchitectureArgs,
};

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 1e-6 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn recurrent_config_derives_checked_dimensions() {
    let config = Qwen35RecurrentConfig::new(1, 2, 2, 3).unwrap();

    assert_eq!(config.key_width(), 2);
    assert_eq!(config.value_width(), 4);
    assert_eq!(config.conv_width(), 8);
    assert_eq!(config.delta_state_elements(), 8);
    assert_eq!(config.conv_state_elements(), 16);

    let err = Qwen35RecurrentConfig::new(2, 3, 4, 3).unwrap_err();
    assert!(err.to_string().contains("value head count"));
}

#[test]
fn causal_conv_step_matches_gguf_kernel_order() {
    let mut state = CausalConv1dState::new(2, 3).unwrap();
    // GGUF shape [kernel, channels]: dimension zero is contiguous, so each
    // channel owns one consecutive kernel row.
    let kernel = [1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
    let mut output = [0.0; 2];

    state.step(&[1.0, 2.0], &kernel, &mut output).unwrap();
    assert_eq!(output, [100.0, 400.0]);

    state.step(&[3.0, 4.0], &kernel, &mut output).unwrap();
    assert_eq!(output, [310.0, 840.0]);

    state.step(&[5.0, 6.0], &kernel, &mut output).unwrap();
    assert_eq!(output, [531.0, 1284.0]);
    assert_eq!(state.history(), &[3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn gated_delta_step_matches_two_token_reference() {
    let config = Qwen35RecurrentConfig::new(1, 1, 2, 1).unwrap();
    let mut state = Qwen35RecurrentState::new(config).unwrap();
    let mut output = [0.0; 2];
    let inv_sqrt_two = 1.0 / 2.0f32.sqrt();

    state
        .gated_delta_step(
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[2.0, 4.0],
            &[0.5f32.ln()],
            &[0.5],
            &mut output,
        )
        .unwrap();
    assert_close(output[0], inv_sqrt_two);
    assert_close(output[1], 2.0 * inv_sqrt_two);
    assert_eq!(state.delta_state(), &[1.0, 0.0, 2.0, 0.0]);

    state
        .gated_delta_step(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[6.0, 8.0],
            &[0.5f32.ln()],
            &[0.25],
            &mut output,
        )
        .unwrap();
    assert_close(output[0], 1.5 * inv_sqrt_two);
    assert_close(output[1], 2.0 * inv_sqrt_two);
    assert_eq!(state.delta_state(), &[0.5, 1.5, 1.0, 2.0]);
}

#[test]
fn gated_delta_uses_gguf_tiled_value_head_order() {
    let config = Qwen35RecurrentConfig::new(2, 4, 1, 1).unwrap();
    let mut state = Qwen35RecurrentState::new(config).unwrap();
    let mut output = [0.0; 4];

    state
        .gated_delta_step(
            &[1.0, 2.0],
            &[1.0, 1.0],
            &[10.0, 20.0, 30.0, 40.0],
            &[0.0; 4],
            &[1.0; 4],
            &mut output,
        )
        .unwrap();

    assert_eq!(output, [10.0, 40.0, 30.0, 80.0]);
}

#[test]
fn recurrent_l2_normalization_adds_epsilon_inside_square_root() {
    let mut values = [3.0, 4.0];

    l2_normalize(&mut values, 9.0);

    let denominator = 34.0f32.sqrt();
    assert_close(values[0], 3.0 / denominator);
    assert_close(values[1], 4.0 / denominator);
}

#[test]
fn recurrent_step_applies_conv_delta_norm_and_output_gate() {
    let config = Qwen35RecurrentConfig::new(1, 1, 1, 1).unwrap();
    let mut state = Qwen35RecurrentState::new(config).unwrap();
    let inputs = Qwen35RecurrentInputs {
        qkv: &[1.0, 2.0, 3.0],
        z: &[1.0],
        beta: &[0.0],
        alpha: &[0.0],
    };
    let weights = Qwen35RecurrentWeights {
        conv: &[1.0, 1.0, 1.0],
        dt_bias: &[0.0],
        decay: &[-1.0],
        norm: &[2.0],
    };
    let mut output = [0.0];

    qwen35_recurrent_step(&mut state, inputs, weights, 0.0, &mut output).unwrap();

    let silu_one = 1.0 / (1.0 + (-1.0f32).exp());
    let silu_three = 3.0 / (1.0 + (-3.0f32).exp());
    assert_close(output[0], 2.0 * silu_one);
    assert_close(state.delta_state()[0], 0.5 * silu_three);

    state.clear();
    assert!(state.delta_state().iter().all(|value| *value == 0.0));
}

#[test]
fn attention_config_derives_checked_dimensions() {
    let config = Qwen35AttentionConfig::new(4, 2, 8, 16).unwrap();

    assert_eq!(config.query_width(), 32);
    assert_eq!(config.kv_width(), 16);
    assert_eq!(config.max_sequence_length(), 16);

    let err = Qwen35AttentionConfig::new(3, 2, 8, 16).unwrap_err();
    assert!(err.to_string().contains("query head count"));
}

#[test]
fn gated_attention_step_matches_two_token_gqa_reference() {
    let config = Qwen35AttentionConfig::new(2, 1, 2, 4).unwrap();
    let mut state = Qwen35AttentionState::new(config);
    let mut output = [0.0; 4];

    state
        .step(
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0],
            &[2.0, 4.0],
            &[0.0; 4],
            &mut output,
        )
        .unwrap();
    assert_eq!(output, [1.0, 2.0, 1.0, 2.0]);

    state
        .step(
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 1.0],
            &[6.0, 8.0],
            &[0.0; 4],
            &mut output,
        )
        .unwrap();

    let scaled = 1.0 / 2.0f32.sqrt();
    let old_for_head_zero = scaled.exp() / (scaled.exp() + 1.0);
    let old_for_head_one = 1.0 / (1.0 + scaled.exp());
    assert_close(
        output[0],
        0.5 * (2.0 * old_for_head_zero + 6.0 * (1.0 - old_for_head_zero)),
    );
    assert_close(
        output[1],
        0.5 * (4.0 * old_for_head_zero + 8.0 * (1.0 - old_for_head_zero)),
    );
    assert_close(
        output[2],
        0.5 * (2.0 * old_for_head_one + 6.0 * (1.0 - old_for_head_one)),
    );
    assert_close(
        output[3],
        0.5 * (4.0 * old_for_head_one + 8.0 * (1.0 - old_for_head_one)),
    );
    assert_eq!(state.len(), 2);
}

#[test]
fn attention_state_enforces_context_limit_without_mutation() {
    let config = Qwen35AttentionConfig::new(1, 1, 2, 1).unwrap();
    let mut state = Qwen35AttentionState::new(config);
    let mut output = [0.0; 2];

    state
        .step(
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[2.0, 4.0],
            &[0.0; 2],
            &mut output,
        )
        .unwrap();
    let err = state
        .step(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[6.0, 8.0],
            &[0.0; 2],
            &mut output,
        )
        .unwrap_err();

    assert!(err.to_string().contains("context limit"));
    assert_eq!(state.len(), 1);
}

#[test]
fn model_state_follows_typed_hybrid_layer_plan() {
    let architecture = Qwen35Architecture::new(Qwen35ArchitectureArgs {
        total_layers: 5,
        attention_key_length: 2,
        attention_value_length: 2,
        rope_sections: [1, 0, 0, 0],
        ssm_conv_kernel: 2,
        ssm_state_size: 2,
        ssm_group_count: 1,
        ssm_time_step_rank: 1,
        ssm_inner_size: 2,
        nextn_predict_layers: 1,
        recurrent_layers: None,
        full_attention_interval: Some(2),
    })
    .unwrap();
    let meta = GgufMeta {
        architecture: GgufArchitecture::Qwen35(architecture),
        n_layers: 5,
        n_embd: 4,
        n_heads: 2,
        n_kv_heads: 1,
        context_length: 16,
        vocab_size: 8,
        bos_token_id: 1,
        eos_token_id: 2,
        n_ff: 8,
        norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_dim: Some(2),
        chat_template: None,
        vocab_tokens: Vec::new(),
        vocab_scores: Vec::new(),
        vocab_types: Vec::new(),
        tensor_data_offset: 0,
        tensors: std::collections::HashMap::new(),
    };

    let state = Qwen35ModelState::from_metadata(&meta, 8).unwrap();

    assert!(matches!(
        state.layer(0),
        Some(Qwen35LayerState::Recurrent(_))
    ));
    assert!(matches!(
        state.layer(1),
        Some(Qwen35LayerState::FullAttention(_))
    ));
    assert!(matches!(
        state.layer(2),
        Some(Qwen35LayerState::Recurrent(_))
    ));
    assert!(matches!(
        state.layer(3),
        Some(Qwen35LayerState::FullAttention(_))
    ));
    assert!(matches!(state.layer(4), Some(Qwen35LayerState::Mtp(_))));
    assert_eq!(state.len(), 5);
}

#[test]
fn interleaved_mrope_matches_qwen35_pair_mapping() {
    let mrope = Qwen35MropeConfig::new([2, 1, 1, 0], 8, 16.0).unwrap();
    let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let original = values;

    mrope.apply(&mut values, 1, 10, [1, 2, 3, 4]).unwrap();

    let angles = [1.0f32, 1.0, 0.75, 0.125];
    for pair in 0..4 {
        let (sin, cos) = angles[pair].sin_cos();
        assert_close(
            values[pair],
            original[pair] * cos - original[pair + 4] * sin,
        );
        assert_close(
            values[pair + 4],
            original[pair] * sin + original[pair + 4] * cos,
        );
    }
    assert_eq!(&values[8..], &[9.0, 10.0]);
}

#[test]
fn attention_preprocess_splits_q_gate_and_applies_head_norm() {
    let config = Qwen35AttentionConfig::new(2, 1, 4, 8).unwrap();
    let mrope = Qwen35MropeConfig::new([1, 1, 0, 0], 4, 10_000.0).unwrap();
    let q_gate = [
        2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 3.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let k_projection = [4.0, 0.0, 0.0, 0.0];
    let norm = [1.0; 4];
    let mut q = [0.0; 8];
    let mut k = [0.0; 4];
    let mut gate = [0.0; 8];

    qwen35_prepare_attention(
        config,
        &mrope,
        [0, 0, 0, 0],
        Qwen35AttentionProjectionInputs {
            q_gate: &q_gate,
            k: &k_projection,
        },
        Qwen35AttentionProjectionWeights {
            q_norm: &norm,
            k_norm: &norm,
        },
        0.0,
        Qwen35AttentionProjectionOutputs {
            q: &mut q,
            k: &mut k,
            gate: &mut gate,
        },
    )
    .unwrap();

    for (actual, expected) in q.iter().zip([2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]) {
        assert_close(*actual, expected);
    }
    for (actual, expected) in k.iter().zip([2.0, 0.0, 0.0, 0.0]) {
        assert_close(*actual, expected);
    }
    assert_eq!(gate, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}
