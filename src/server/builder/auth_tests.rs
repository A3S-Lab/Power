use super::PowerServerBuilder;
use crate::{config::PowerConfig, error::PowerError, server::auth::ApiKeyAuth};
use std::sync::Arc;

#[tokio::test]
async fn injected_auth_is_retained_and_ambiguous_static_keys_fail_before_startup() {
    let provider = Arc::new(ApiKeyAuth::new(&["hermetic-test-token".into()]));
    let options = PowerServerBuilder::new(PowerConfig::default())
        .with_auth_provider(provider.clone())
        .into_options();
    assert!(options
        .auth_provider
        .unwrap()
        .authenticate("hermetic-test-token"));
    let mut config = PowerConfig::default();
    config.api_keys = vec!["a-different-hermetic-test-token".into()];
    assert!(
        matches!(PowerServerBuilder::new(config).with_auth_provider(provider).start().await,
        Err(PowerError::Config(message)) if message.contains("mutually exclusive"))
    );
}
