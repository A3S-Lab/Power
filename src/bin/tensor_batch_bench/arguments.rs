use std::path::PathBuf;

use a3s_power::error::{PowerError, Result};

pub(super) struct Arguments {
    values: Vec<String>,
}

impl Arguments {
    pub(super) fn new(values: Vec<String>) -> Self {
        Self { values }
    }

    pub(super) fn required(&mut self, name: &str) -> Result<String> {
        self.optional(name)?
            .ok_or_else(|| PowerError::InvalidRequest(format!("missing required argument {name}")))
    }

    pub(super) fn required_path(&mut self, name: &str) -> Result<PathBuf> {
        self.required(name).map(PathBuf::from)
    }

    pub(super) fn optional_path(&mut self, name: &str) -> Result<Option<PathBuf>> {
        self.optional(name).map(|value| value.map(PathBuf::from))
    }

    pub(super) fn required_number<T>(&mut self, name: &str) -> Result<T>
    where
        T: std::str::FromStr,
    {
        parse_number(name, &self.required(name)?)
    }

    pub(super) fn optional_number<T>(&mut self, name: &str) -> Result<Option<T>>
    where
        T: std::str::FromStr,
    {
        self.optional(name)?
            .map(|value| parse_number(name, &value))
            .transpose()
    }

    pub(super) fn optional(&mut self, name: &str) -> Result<Option<String>> {
        let Some(index) = self.values.iter().position(|value| value == name) else {
            return Ok(None);
        };
        if index.saturating_add(1) >= self.values.len() || self.values[index + 1].starts_with("--")
        {
            return Err(PowerError::InvalidRequest(format!(
                "argument {name} requires a value"
            )));
        }
        self.values.remove(index);
        Ok(Some(self.values.remove(index)))
    }

    pub(super) fn finish(self) -> Result<()> {
        self.ensure_empty()
    }

    pub(super) fn ensure_empty(&self) -> Result<()> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(PowerError::InvalidRequest(format!(
                "unknown tensor batch benchmark argument '{}'",
                self.values[0]
            )))
        }
    }
}

fn parse_number<T>(name: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| PowerError::InvalidRequest(format!("argument {name} must be a valid number")))
}
