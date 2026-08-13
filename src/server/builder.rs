use std::sync::Arc;

use crate::backend::{Backend, BackendRegistry};
use crate::config::PowerConfig;
use crate::error::Result;

use super::log_stream::LogBuffer;

/// Typed composition root for the Power HTTP server.
///
/// Injected backends are registered before Power's built-in backends and
/// therefore have higher priority. Call [`Self::without_default_backends`] when
/// the caller supplies the complete backend set.
pub struct PowerServerBuilder {
    options: PowerServerOptions,
}

pub(super) struct PowerServerOptions {
    pub(super) config: PowerConfig,
    pub(super) log_buffer: Option<LogBuffer>,
    pub(super) backends: BackendRegistry,
    pub(super) include_default_backends: bool,
}

impl PowerServerBuilder {
    pub fn new(config: PowerConfig) -> Self {
        Self {
            options: PowerServerOptions {
                config,
                log_buffer: None,
                backends: BackendRegistry::new(),
                include_default_backends: true,
            },
        }
    }

    pub fn with_log_buffer(mut self, log_buffer: LogBuffer) -> Self {
        self.options.log_buffer = Some(log_buffer);
        self
    }

    /// Register a typed backend ahead of the built-in backends.
    pub fn with_backend(mut self, backend: Arc<dyn Backend>) -> Self {
        self.options.backends.register(backend);
        self
    }

    /// Replace the currently injected backend set.
    ///
    /// Built-in backends are still appended unless
    /// [`Self::without_default_backends`] is also selected.
    pub fn with_backend_registry(mut self, backends: BackendRegistry) -> Self {
        self.options.backends = backends;
        self
    }

    /// Start with only caller-injected backends.
    pub fn without_default_backends(mut self) -> Self {
        self.options.include_default_backends = false;
        self
    }

    pub async fn start(self) -> Result<()> {
        super::start_with_options(self.into_options()).await
    }

    pub(super) fn into_options(self) -> PowerServerOptions {
        self.options
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backend::test_utils::MockBackend;
    use crate::config::PowerConfig;

    use super::PowerServerBuilder;

    #[test]
    fn injected_backends_are_typed_and_keep_registration_order() {
        let options = PowerServerBuilder::new(PowerConfig::default())
            .with_backend(Arc::new(
                MockBackend::success()
                    .with_name("olmoe")
                    .with_family("olmoe"),
            ))
            .with_backend(Arc::new(MockBackend::success().with_name("fallback")))
            .into_options();

        assert_eq!(options.backends.list_names(), ["olmoe", "fallback"]);
        assert!(options.include_default_backends);
    }

    #[test]
    fn default_backends_can_be_disabled_explicitly() {
        let options = PowerServerBuilder::new(PowerConfig::default())
            .without_default_backends()
            .into_options();

        assert!(!options.include_default_backends);
        assert!(options.backends.list_names().is_empty());
    }

    #[test]
    fn builder_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PowerServerBuilder>();
    }
}
