//! Model-neutral shared sessions and exclusive mutable replicas.

mod pool;
mod replica;
mod slot;
mod types;

pub use pool::{ModelSession, ModelSessionPool};
pub use replica::ModelSessionReplica;
pub use types::{
    ModelSessionBinding, ModelSessionPoolPolicy, ModelSessionPoolSnapshot, ModelSessionSpec,
};
