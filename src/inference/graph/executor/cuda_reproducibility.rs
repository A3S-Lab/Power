/// Maximum leading-axis items included in one CUDA reduction launch when the
/// arithmetic is otherwise independent per item.
///
/// A fixed execution quantum prevents later batch items from changing an
/// earlier item's cuBLAS reduction kernel. The bound is topology-neutral and
/// also caps temporary working sets. Raising it requires generic bit-prefix
/// parity evidence across the old boundary.
pub(super) const REPRODUCIBLE_BATCH_ITEMS: usize = 32;
