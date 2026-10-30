//! Family-neutral model execution contracts and implementations.

pub mod assessment;
mod completion;
mod device_resources;
mod device_table;
mod domain;
mod error;
mod execution_path;
mod host_tables;
mod import_transforms;
mod inventory;
mod kernel_cache;
mod lanes;
pub mod memory;
mod native;
#[cfg(test)]
mod ngram_steps_tests;
mod operation;
mod operators;
mod planning;
pub mod programs;
mod progressive;
mod slot_directory;
mod stored_table;
mod streamed_table;
pub use programs::{
    CommitSpan, CompletedHeadWork, CompletedImportWork, CompletedStateWork, CompletedTargetWork,
    CompletedVisionWork, GraphError, HeadProgram, ImportProgram, PlannedSealedGraphs,
    PreparedDraftGraphs, PreparedDrafterGraphs, PreparedHeadGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, SealReport,
    SealedGraphsHeap, StateProgram, TargetOutput, TargetProgram, VisionProgram,
};
pub mod platform;
mod residency;
mod resident_weights;
mod resources;

pub use completion::{Completed, Completion, CompletionWake};
pub use device_resources::{
    ConditioningRange, ConditioningRef, FeatureRef, ImageRef, LogitsRef, ResourceDomain,
    ResourceError, WeakFeatureRef,
};
pub use domain::{
    require_normal, ClaimRefusal, DeviceHeap, DomainError, DomainRequirements, DomainReservation,
    ExecutorDomain, HeadFlight, MemoryChargeReconciliation, NativeFamily, OpenRequirements,
    PendingOperationOutcome, PhysicalDecision, PrimingCompletionFlight, ProgramFamily,
    ReservedResources, ResumeState, StateBindings, StoreAllocation, SubmitFailure,
    TargetContinuation, TargetFlight, TargetHostTiming, VisionFlight,
};
pub use error::{
    CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError,
    TableReadError,
};
pub use execution_path::ExecutionPath;
pub use inventory::kernel_inventory;
pub use kernel_cache::{
    KernelCache, KernelCacheError, StoredEvidence, TuningCacheKey, DEFAULT_KERNEL_CACHE_BYTES,
};
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadConditioning, HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs,
    ResidentWeightSlot, StateLaunchCore, StateLaunchInputs, StateWork, TargetHistoryRead,
    TargetLaunchCore, TargetLaunchInputs, TargetLaunchWorkspace, TargetReads, TargetTokens,
    ValidatedHeadLaunch, ValidatedImportLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchCore, VisionLaunchInputs,
};
pub use magnitude_batching::{self as batching, Demand, LaunchClass};
/// Development-only tuning pin for bit-exact measurement runs
/// (`magnitude-harness --tuning-record`); see the module documentation.
#[cfg(feature = "pinned-tuning")]
pub use native::pinned_tuning;
/// Development-only tuning survey for replaying the search
/// (`magnitude-harness --tuning-survey`); see the module documentation.
#[cfg(feature = "tuning-survey")]
pub use native::tuning_survey;
pub use native::{
    attention_points, row_points, AttestedPrograms, CatalogError, CatalogFailure, NumericalProfile,
    PinnedChoice, PointShape, QualificationCase, QualificationReport, Receipt,
    ResolvedNumericalProfile, SearchProgress, TracedWork, TunedEntry, TuningContext, TuningEvent,
    TuningLimits, TuningObserver, TuningOrigin, TuningRoom, TuningSettings, TuningTrace,
    TuningWeightSource, UnpreparedUnits, UnreportedTuning, ZeroTuningWeights, MIN_SAMPLE_SECONDS,
    ROTATION_LAYERS, SEARCH_SETTINGS, SEARCH_VERSION, STANDARD_TUNING_BUDGET, TRACE_SCHEMA,
    TUNING_CONTEXTS, TUNING_ROWS,
};
pub use operation::{
    CommittedClass, DraftForm, ExecutableKind, FeatureReader, FeatureRows, FeatureSpan, GroupKey,
    HeadPhase, Operation, OperationError, Outcome, Priming, ProgramIdentity, RequestId,
    ResourceDomainId, RowResult, Sampling, SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use operators::routed::{
    Expansion, ExpertScales, GeneralRoutedBinding, GeneralRoutedScales, GeneralRoutedShape,
};
pub use operators::short_conv::{ShortConvBinding, ShortConvShape};
pub use operators::state_space::{StateSpaceBinding, StateSpaceScales, StateSpaceShape};
pub use operators::vision::{VisionEntry, VisionKernel};
pub use planning::{
    draft_ladder, image_cell_limit, max_image_cells, resident_element, resident_elements,
    resident_layout, source_element, ArtifactComponent, ArtifactComponentKind, AssessmentFit,
    AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms, AttentionBinding,
    AttentionShape, BackendPlan, CapabilityPlan, ComponentPlan, ComponentSelection,
    ConvolutionBinding, ConvolvedLayerBinding, CorrectorBinding, CorrectorLayerBinding,
    DenseBinding, DenseBranchBinding, DenseScales, DeviceProcessBytes, DraftBlockBinding,
    DraftProgramPlan, DraftWidths, EmbeddingBinding, ExecutionPlan, ExecutionPlanDraft,
    ExecutionPlanner, FeaturesBinding, FeedForwardProgramSlot, GraphSlots, HeadBinding,
    HeadProgramPlan, HostProcessBytes, HostTablePlan, ImportProgramSlot, MarkovBinding,
    MixerProgramSlot, ModelLoadPlan, NGramBinding, NativeGraphCharge, ParallelBinding,
    PerLayerBinding, PerLayerEntryBinding, PlannedDevice, PlannedMethod, ProgramPlan,
    ReadoutBinding, ReadoutClassChoice, ReadoutHead, RecurrentBinding, ResolvedPolicy,
    ResourceBytes, ResourceCapacity, ResourceLimits, ResourcePlan, ResourcePlanner, RoutedBinding,
    ScalableWeight, SealedGraphHostBytes, SelectionBinding, SelectionShape, SelectorBinding,
    SharedHeadProgramPlan, StartupSlots, StateCapacityPlan, StateProgramPlan, StateResourcePlan,
    StateStorePlan, SublayerTail, TapProgramPlan, TargetBlockProgramSlot, TargetProgramPlan,
    TensorOperations, VisionProgramPlan, WeightPlan, WeightScalePlan, WeightStorageIdentity,
    CONVERSATION_CHECKPOINTS, CPU_IMAGE_CELLS, MAX_DRAFT_PROPOSALS, MAX_IMAGE_CELLS,
};

/// Whether this executor runs `definition` with its draft head selected.
pub fn head_admitted(definition: &magnitude_family_contracts::ModelDefinition) -> bool {
    definition.head.is_some() && operators::admit(definition, true).is_ok()
}
pub use residency::{
    ComponentLoader, ImportArtifactTensor, ResidentWeight, Stored, StoredTensor, WeightImportError,
};
pub use residency::{MappedImportReport, ResidencyStore};
pub use resident_weights::{
    ResidencyError, ResidentHead, ResidentOutput, ResidentPlanes, ResidentRoles, ResidentTarget,
    ResidentVision,
};
pub use resources::{
    AllocatedResources, AllocationError, GraphOutputOwner, GraphOutputTensor, ImportWorkspaceLease,
    NativeGraphOutputLease, NativeGraphPool, NativeGraphWorkspaceLease, PoolClass,
    ResourceAllocator, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
    VisionGraphPool,
};
