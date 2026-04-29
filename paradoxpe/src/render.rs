#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderTransformSample {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

impl Default for RenderTransformSample {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Target surface for exporting physics-owned transforms into a render-visible buffer.
///
/// ParadoxPE stays scheduler-agnostic by writing into this trait instead of depending on MPS
/// storage types directly. Integration layers such as `tl-core` can provide concrete adapters.
pub trait RenderTransformTarget {
    fn capacity(&self) -> usize;
    fn write_transform(&self, index: usize, sample: RenderTransformSample);
}
