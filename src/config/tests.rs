fn valid_model_signing_key_hex() -> String {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
    hex::encode(signing_key.verifying_key().to_bytes())
}

mod basic;
mod policy;
mod validation;
