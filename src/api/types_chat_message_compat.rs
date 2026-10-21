//! OpenAI request-only compatibility for completed assistant tool histories.
//!
//! Keep the public message/content types strict. The existing typed message
//! deserializer still sees every field, including duplicate fields and unknown
//! policy, while this adapter can represent a nullable content value temporarily.
use super::ChatCompletionMessage;
use crate::backend::types::MessageContent;
use serde::de::{DeserializeSeed, Error, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;

pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<ChatCompletionMessage>, D::Error>
where
    D: Deserializer<'de>,
{
    Vec::<RequestMessage>::deserialize(deserializer)
        .map(|messages| messages.into_iter().map(|message| message.0).collect())
}

struct RequestMessage(ChatCompletionMessage);

impl<'de> Deserialize<'de> for RequestMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RequestMessageVisitor)
    }
}

struct RequestMessageVisitor;

impl<'de> Visitor<'de> for RequestMessageVisitor {
    type Value = RequestMessage;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an OpenAI chat message")
    }

    fn visit_map<M>(self, access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut content_present = false;
        let mut content_null = false;
        let message = ChatCompletionMessage::deserialize(
            serde::de::value::MapAccessDeserializer::new(NullableContentMap {
                inner: access,
                next_is_content: false,
                content_present: &mut content_present,
                content_null: &mut content_null,
            }),
        )?;
        if !content_present || content_null {
            let completed_calls = message.tool_calls.as_ref().is_some_and(|calls| {
                !calls.is_empty()
                    && calls.iter().all(|call| {
                        !call.id.trim().is_empty()
                            && call.tool_type == "function"
                            && !call.function.name.trim().is_empty()
                    })
            });
            if message.role != "assistant" || !completed_calls {
                return Err(M::Error::custom(
                    "null or missing content requires an assistant message with nonempty completed function tool_calls",
                ));
            }
        }
        Ok(RequestMessage(message))
    }
}

struct NullableContentMap<'a, M> {
    inner: M,
    next_is_content: bool,
    content_present: &'a mut bool,
    content_null: &'a mut bool,
}

impl<'de, M> MapAccess<'de> for NullableContentMap<'_, M>
where
    M: MapAccess<'de>,
{
    type Error = M::Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: DeserializeSeed<'de>,
    {
        self.inner.next_key::<String>()?.map_or(Ok(None), |key| {
            self.next_is_content = key == "content";
            if self.next_is_content {
                *self.content_present = true;
            }
            seed.deserialize(serde::de::value::StringDeserializer::<M::Error>::new(key))
                .map(Some)
        })
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        if !self.next_is_content {
            return self.inner.next_value_seed(seed);
        }
        match self.inner.next_value::<Option<MessageContent>>()? {
            None => {
                *self.content_null = true;
                seed.deserialize(serde::de::value::StrDeserializer::<M::Error>::new(""))
            }
            Some(MessageContent::Text(text)) => {
                seed.deserialize(serde::de::value::StringDeserializer::<M::Error>::new(text))
            }
            Some(content @ MessageContent::Parts(_)) => {
                // Parts have already passed their original typed validation;
                // replay them to the unchanged public message deserializer.
                let value = serde_json::to_value(content).map_err(M::Error::custom)?;
                seed.deserialize(value).map_err(M::Error::custom)
            }
        }
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}
