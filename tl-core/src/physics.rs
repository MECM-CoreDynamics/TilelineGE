//! Physics integration adapters shared across Tileline runtime layers.
//!
//! This module intentionally keeps ParadoxPE decoupled from MPS-owned storage types.
//! `paradoxpe` exports generic render transform targets, while `tl-core` owns the concrete
//! adapters that bridge those targets to MPS buffers.

use mps::{
    DispatcherDoubleBufferedTransforms, DispatcherTransformSample, DoubleBufferedTransformStorage,
    TransformSample,
};
use paradoxpe::{PhysicsWorld, RenderTransformSample, RenderTransformTarget};

pub struct MpsThreadPoolRenderTransformTarget<'a> {
    storage: &'a DoubleBufferedTransformStorage,
    slot: usize,
}

impl<'a> MpsThreadPoolRenderTransformTarget<'a> {
    pub fn new(storage: &'a DoubleBufferedTransformStorage, slot: usize) -> Self {
        Self { storage, slot }
    }
}

impl RenderTransformTarget for MpsThreadPoolRenderTransformTarget<'_> {
    fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    fn write_transform(&self, index: usize, sample: RenderTransformSample) {
        self.storage.write_transform_to_slot(
            self.slot,
            index,
            TransformSample {
                position: sample.position,
                rotation: sample.rotation,
            },
        );
    }
}

pub struct MpsDispatcherRenderTransformTarget<'a> {
    storage: &'a DispatcherDoubleBufferedTransforms,
    slot: usize,
}

impl<'a> MpsDispatcherRenderTransformTarget<'a> {
    pub fn new(storage: &'a DispatcherDoubleBufferedTransforms, slot: usize) -> Self {
        Self { storage, slot }
    }
}

impl RenderTransformTarget for MpsDispatcherRenderTransformTarget<'_> {
    fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    fn write_transform(&self, index: usize, sample: RenderTransformSample) {
        self.storage.write_transform_to_slot(
            self.slot,
            index,
            DispatcherTransformSample {
                position: sample.position,
                rotation: sample.rotation,
            },
        );
    }
}

pub fn write_world_render_transforms_to_thread_pool_storage(
    world: &PhysicsWorld,
    storage: &DoubleBufferedTransformStorage,
    slot: usize,
) -> usize {
    let target = MpsThreadPoolRenderTransformTarget::new(storage, slot);
    world.write_render_transforms(&target)
}

pub fn write_world_render_transforms_to_dispatcher_storage(
    world: &PhysicsWorld,
    storage: &DispatcherDoubleBufferedTransforms,
    slot: usize,
) -> usize {
    let target = MpsDispatcherRenderTransformTarget::new(storage, slot);
    world.write_render_transforms(&target)
}
