//! Model-neutral finite execution-profile contracts.
//!
//! Model integrations own every shape-class meaning. Power only validates
//! opaque identities, aggregate resource envelopes, runtime bindings, and
//! digest-only execution evidence.

mod binding;
mod declaration;
mod digest;
mod evidence;

pub use binding::{DynamicShapeFallback, ShapeProfileBinding};
pub use declaration::{ShapeProfile, ShapeProfileDeclaration, ShapeProfileRequest};
pub use evidence::{
    ShapeProfileExecutionEvidence, ShapeProfileExecutionPath, ShapeProfileFallbackReason,
    ShapeProfileSelection,
};
