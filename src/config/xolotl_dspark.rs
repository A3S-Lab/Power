use super::*;

/// Explicit ownership of the checkpoint's speculative draft. No arbitrary engine environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XolotlDSparkConfig {
    pub sidecar: PathBuf,
    pub sidecar_sha256: String,
    pub depth: u32,
    pub confidence_bp: u32,
    pub evict_prompt_tokens: u32,
}
impl XolotlDSparkConfig {
    pub fn validate(&self) -> Result<()> {
        let path = self
            .sidecar
            .to_str()
            .ok_or_else(|| PowerError::Config("xolotl.dspark sidecar path must be UTF-8".into()))?;
        if !self.sidecar.is_absolute()
            || path.len() > 4096
            || path.contains(['\0', '\r', '\n'])
            || self.sidecar_sha256.len() != 64
            || !self
                .sidecar_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || !(1..=5).contains(&self.depth)
            || self.confidence_bp > 10000
            || self.evict_prompt_tokens > 1048576
            || self.evict_prompt_tokens % 128 != 0
        {
            return Err(PowerError::Config(
                "xolotl.dspark exceeds its admitted path, digest or speculative bounds".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn append_env(&self, env: &mut Vec<(String, String)>) -> Result<()> {
        self.validate()?;
        let path = self
            .sidecar
            .to_str()
            .ok_or_else(|| PowerError::Config("xolotl.dspark sidecar path must be UTF-8".into()))?;
        for (key, value) in [
            ("STRATA_SERVE_DSPARK", path.to_owned()),
            ("STRATA_DSPARK_DEPTH", self.depth.to_string()),
            (
                "STRATA_DSPARK_CONF",
                format!(
                    "{}.{:04}",
                    self.confidence_bp / 10000,
                    self.confidence_bp % 10000
                ),
            ),
            (
                "STRATA_SERVE_DSPARK_EVICT",
                self.evict_prompt_tokens.to_string(),
            ),
        ] {
            env.push((key.into(), value));
        }
        Ok(())
    }
}
#[cfg(test)]
#[path = "xolotl_dspark_tests.rs"]
mod tests;
