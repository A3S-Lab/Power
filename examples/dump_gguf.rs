//! Dump tokenizer merges and more metadata from GGUF.

#[cfg(feature = "picolm")]
fn main() {
    use a3s_power::backend::gguf_stream::GgufFile;

    let path = "/tmp/qwen2.5-0.5b-q4_k_m.gguf";
    let gguf = GgufFile::open(std::path::Path::new(path)).unwrap();
    let meta = &gguf.meta;

    // Check if merges exist by looking at raw metadata
    // Our GgufMeta doesn't parse merges yet, so let's check vocab structure

    // Token type distribution
    let mut type_counts = std::collections::HashMap::new();
    for &t in &meta.vocab_types {
        *type_counts.entry(t).or_insert(0) += 1;
    }
    eprintln!("Token type distribution: {:?}", type_counts);

    // Show tokens around common words to understand encoding
    eprintln!("\n--- Tokens containing 'What' ---");
    for (i, tok) in meta.vocab_tokens.iter().enumerate() {
        if tok.contains("What") && i < 200000 {
            let ttype = meta.vocab_types.get(i).copied().unwrap_or(0);
            eprintln!("  [{i:6}] type={ttype} {:?}", tok);
        }
    }

    // Show Ġ-prefixed tokens (GPT-style space encoding)
    eprintln!("\n--- First 10 Ġ-prefixed tokens ---");
    let mut count = 0;
    for (i, tok) in meta.vocab_tokens.iter().enumerate() {
        if tok.starts_with("Ġ") && count < 10 {
            eprintln!("  [{i:6}] {:?}", tok);
            count += 1;
        }
    }

    // Check byte tokens
    eprintln!("\n--- Byte tokens (<0xHH>) ---");
    count = 0;
    for (i, tok) in meta.vocab_tokens.iter().enumerate() {
        if tok.starts_with("<0x") && count < 5 {
            let ttype = meta.vocab_types.get(i).copied().unwrap_or(0);
            eprintln!("  [{i:6}] type={ttype} {:?}", tok);
            count += 1;
        }
    }

    // Show BOS/EOS area
    eprintln!("\n--- Tokens around BOS ({}) ---", meta.bos_token_id);
    let bos = meta.bos_token_id as usize;
    for i in bos.saturating_sub(2)..=(bos + 2).min(meta.vocab_tokens.len() - 1) {
        let ttype = meta.vocab_types.get(i).copied().unwrap_or(0);
        eprintln!("  [{i:6}] type={ttype} {:?}", &meta.vocab_tokens[i]);
    }

    // Test: what tokens would "What is 2+2?" map to with simple lookup?
    eprintln!("\n--- Manual token lookup for 'What is 2+2?' ---");
    let text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    let mut pos = 0;
    let mut token_ids = Vec::new();
    while pos < text.len() {
        let remaining = &text[pos..];
        // Try longest match
        let mut best_len = 0;
        let mut best_id = 0;
        for (id, tok) in meta.vocab_tokens.iter().enumerate() {
            if remaining.starts_with(tok.as_str()) && tok.len() > best_len {
                best_len = tok.len();
                best_id = id;
            }
        }
        if best_len > 0 {
            token_ids.push(best_id);
            pos += best_len;
        } else {
            // Single byte fallback
            eprintln!(
                "  No match at pos {pos}: {:?}",
                &remaining[..1.min(remaining.len())]
            );
            pos += 1;
        }
    }
    eprintln!("  Token IDs: {:?}", token_ids);
    for &id in &token_ids {
        eprintln!("  [{id:6}] {:?}", &meta.vocab_tokens[id]);
    }
}

#[cfg(not(feature = "picolm"))]
fn main() {
    eprintln!("Requires --features picolm");
}
