//! Async physics step runner via MPS (Multi-Processing Scaler).
//!
//! The runner now targets the bare-metal `TaskDispatcher` path so runtime
//! overlap semantics match the low-latency MPS frame pipeline. The world stays
//! behind `Arc<Mutex<_>>` for now so the rest of TLApp can keep its current
//! synchronous borrow flow while we migrate internal world phases away from the
//! remaining single-step hot path.

use mps::{
    DispatcherPhaseCallbacks, DispatcherPhasePlan, MpsThreadPoolMetrics, PhysicsDispatchTrigger,
    TaskDispatcher, TaskDispatcherConfig,
};
use paradoxpe::PhysicsWorld;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::Duration;
use tl_core::write_world_render_transforms_to_dispatcher_storage;

/// MPS dispatcher tuning for the physics runner.
///
/// Controls queue depth and per-phase dispatch plans. Use [`PhysicsMpsRunnerConfig::standard`]
/// for normal operation and [`PhysicsMpsRunnerConfig::heimdall`] for the ultra-aggressive
/// Heimdall profile (higher power and thermal draw; document the tradeoff to users).
#[derive(Debug, Clone)]
pub struct PhysicsMpsRunnerConfig {
    /// Lock-free ring capacity for MPS task queue slots.
    pub queue_capacity: usize,
    /// MPS dispatch plan for the broadphase phase.
    pub broadphase_plan: DispatcherPhasePlan,
    /// MPS dispatch plan for the narrowphase phase.
    pub narrowphase_plan: DispatcherPhasePlan,
    /// MPS dispatch plan for the solver phase.
    pub solver_plan: DispatcherPhasePlan,
    /// MPS dispatch plan for the integrate phase.
    pub integrate_plan: DispatcherPhasePlan,
}

impl Default for PhysicsMpsRunnerConfig {
    fn default() -> Self {
        Self::standard(num_cpus::get())
    }
}

impl PhysicsMpsRunnerConfig {
    /// Standard config for desktop/Mac builds. All phases get a minimal (1,1) plan so MPS
    /// can track and report them. Queue capacity matches the established baseline.
    pub fn standard(_logical_threads: usize) -> Self {
        Self {
            queue_capacity: 262_144,
            broadphase_plan: DispatcherPhasePlan::new(1, 1),
            narrowphase_plan: DispatcherPhasePlan::new(1, 1),
            solver_plan: DispatcherPhasePlan::new(1, 1),
            integrate_plan: DispatcherPhasePlan::new(1, 1),
        }
    }

    /// Heimdall ultra-aggressive profile: doubled queue depth and scaled phase chunk counts
    /// based on available logical threads.
    ///
    /// **Tradeoff**: higher CPU utilization ceiling, more power draw, potentially louder
    /// cooling. Never the silent default — only activated via explicit `--tick-profile heimdall`.
    pub fn heimdall(logical_threads: usize) -> Self {
        let extra = (logical_threads / 8).max(1);
        Self {
            queue_capacity: 524_288,
            broadphase_plan: DispatcherPhasePlan::new(extra, 1),
            narrowphase_plan: DispatcherPhasePlan::new(extra, 1),
            solver_plan: DispatcherPhasePlan::new(extra, 1),
            integrate_plan: DispatcherPhasePlan::new((extra + 1).max(2), 1),
        }
    }
}

/// Completion handle for an async physics step.
///
/// Call [`wait`] before reading or writing the `PhysicsWorld` again.
///
/// [`wait`]: PhysicsStepToken::wait
pub struct PhysicsStepToken {
    rx: mpsc::Receiver<u32>,
    dispatcher: Arc<TaskDispatcher>,
    frame_id: u64,
    await_publish: bool,
}

impl PhysicsStepToken {
    /// Block until the physics step completes and return the substep count.
    pub fn wait(self) -> u32 {
        let substeps = self.rx.recv().unwrap_or(0);
        if self.await_publish
            && !self
                .dispatcher
                .wait_for_completed_frame(self.frame_id, Duration::from_millis(250))
        {
            eprintln!(
                "[physics mps] frame {} finished stepping but did not publish within 250 ms",
                self.frame_id
            );
        }
        substeps
    }
}

/// Wraps a `PhysicsWorld` for async MPS-based stepping.
///
/// All non-step access happens synchronously via [`borrow`] / [`borrow_mut`].
/// [`step_begin`] triggers a one-frame physics job on the custom MPS pool and
/// returns immediately. The caller must call [`PhysicsStepToken::wait`] before
/// touching the world again.
///
/// [`borrow`]: PhysicsMpsRunner::borrow
/// [`borrow_mut`]: PhysicsMpsRunner::borrow_mut
/// [`step_begin`]: PhysicsMpsRunner::step_begin
pub struct PhysicsMpsRunner {
    world: Arc<Mutex<PhysicsWorld>>,
    dispatcher: Arc<TaskDispatcher>,
    next_frame_id: AtomicU64,
    config: PhysicsMpsRunnerConfig,
}

impl PhysicsMpsRunner {
    /// Create a runner from an existing world using the standard config.
    pub fn new(world: PhysicsWorld) -> Self {
        Self::new_with_config(world, PhysicsMpsRunnerConfig::default())
    }

    /// Create a runner from an existing world with explicit tuning config.
    pub fn new_with_config(world: PhysicsWorld, config: PhysicsMpsRunnerConfig) -> Self {
        let mut dispatcher_config = TaskDispatcherConfig::default();
        dispatcher_config.queue_capacity =
            dispatcher_config.queue_capacity.max(config.queue_capacity);
        dispatcher_config.transform_capacity =
            dispatcher_config.transform_capacity.max(world.body_count());
        let dispatcher = Arc::new(
            TaskDispatcher::new(dispatcher_config)
                .expect("failed to create bare-metal MPS task dispatcher"),
        );
        let transforms = dispatcher.transforms();
        write_world_render_transforms_to_dispatcher_storage(
            &world,
            transforms.as_ref(),
            transforms.render_read_slot(),
        );
        Self {
            world: Arc::new(Mutex::new(world)),
            dispatcher,
            next_frame_id: AtomicU64::new(1),
            config,
        }
    }

    /// Borrow the world. Blocks only if a step is currently in flight.
    #[inline]
    pub fn borrow(&self) -> MutexGuard<'_, PhysicsWorld> {
        self.world.lock().unwrap()
    }

    /// Mutably borrow the world. Blocks only if a step is currently in flight.
    #[inline]
    pub fn borrow_mut(&self) -> MutexGuard<'_, PhysicsWorld> {
        self.world.lock().unwrap()
    }

    /// Replace the inner world (e.g. for simulation reset).
    ///
    /// The caller must drain any pending [`PhysicsStepToken`] first.
    pub fn replace(&self, new_world: PhysicsWorld) {
        let mut guard = self.world.lock().unwrap();
        *guard = new_world;
        let transforms = self.dispatcher.transforms();
        write_world_render_transforms_to_dispatcher_storage(
            &guard,
            transforms.as_ref(),
            transforms.render_read_slot(),
        );
    }

    /// Frame ID of the physics step currently executing on the MPS dispatcher, if any.
    ///
    /// Compare with the render frame ID to measure Render-N / Physics-N+1 overlap depth.
    pub fn in_flight_frame_id(&self) -> Option<u64> {
        self.dispatcher.metrics().active_frame_id
    }

    /// Number of physics frames that have been submitted ahead of `render_frame_id`.
    ///
    /// A value of 1 means Physics N+1 is already in flight while Render N is being composed —
    /// the ideal overlap state. Values > 1 indicate the physics dispatcher is running ahead;
    /// values of 0 mean physics and render are not overlapping this frame.
    pub fn lag_frame_count(&self, render_frame_id: u64) -> u64 {
        self.next_frame_id
            .load(Ordering::Relaxed)
            .saturating_sub(render_frame_id + 1)
    }

    /// Snapshot runtime-visible metrics from the bare-metal dispatcher.
    pub fn thread_pool_metrics(&self) -> MpsThreadPoolMetrics {
        let metrics = self.dispatcher.metrics();
        MpsThreadPoolMetrics {
            worker_count: metrics.worker_count,
            queued_jobs: metrics.queued_jobs,
            in_flight_jobs: metrics.in_flight_jobs,
            completed_jobs: metrics.completed_jobs,
            completed_frames: metrics.published_frames,
            latest_completed_frame: metrics.latest_published_frame,
            active_frame_id: metrics.active_frame_id,
            simd_backend: metrics.simd_backend,
            simd_lanes: metrics.simd_lanes,
            phase_jobs: metrics.phase_jobs,
            phase_completed_jobs: metrics.phase_completed_jobs,
            hot_worker_ratio: metrics.hot_worker_ratio,
            phase_skew: metrics.phase_skew,
            queue_saturation_events: metrics.queue_saturation_events,
        }
    }

    /// Submit `world.step(dt)` on the bare-metal MPS dispatcher.
    ///
    /// Returns immediately. Call [`PhysicsStepToken::wait`] before touching
    /// the world again.
    pub fn step_begin(&self, dt: f32) -> PhysicsStepToken {
        let (tx, rx) = mpsc::sync_channel(1);
        let fallback_tx = tx.clone();
        let world = Arc::clone(&self.world);
        let frame_id = self.next_frame_id.fetch_add(1, Ordering::Relaxed);
        let planned_step = {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut world = world.lock().unwrap();
                world.prepare_step_execution(dt)
            }));
            match result {
                Ok(plan) => plan,
                Err(_) => {
                    eprintln!(
                        "[physics mps] frame {} panicked while preparing step execution plan",
                        frame_id
                    );
                    None
                }
            }
        };

        let Some(plan) = planned_step else {
            let _ = fallback_tx.send(0);
            return PhysicsStepToken {
                rx,
                dispatcher: Arc::clone(&self.dispatcher),
                frame_id,
                await_publish: false,
            };
        };

        let integration_world = Arc::clone(&world);
        let integration_tx = tx;
        let integration = Arc::new(move |ctx: &mps::DispatcherTaskContext| {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut world = integration_world.lock().unwrap();
                let completed_substeps = world.step_with_execution_plan(plan);
                write_world_render_transforms_to_dispatcher_storage(
                    &world,
                    ctx.transforms.as_ref(),
                    ctx.physics_write_slot,
                );
                completed_substeps
            }));

            match result {
                Ok(completed_substeps) => {
                    let _ = integration_tx.send(completed_substeps);
                }
                Err(_) => {
                    eprintln!(
                        "[physics mps] frame {} panicked during planned step execution; returning 0 substeps",
                        ctx.frame_id
                    );
                    let _ = integration_tx.send(0);
                }
            }
        });

        let callbacks = DispatcherPhaseCallbacks::default().with_integration(integration);
        let trigger = PhysicsDispatchTrigger::with_phase_plans(
            frame_id,
            self.config.broadphase_plan,
            self.config.narrowphase_plan,
            self.config.solver_plan,
            self.config.integrate_plan,
            DispatcherPhasePlan::default(),
        );
        let await_publish =
            if let Err(err) = self.dispatcher.trigger_next_physics(trigger, callbacks) {
                eprintln!(
                    "[physics mps] failed to trigger physics frame {}: {:?}",
                    frame_id, err
                );
                let _ = fallback_tx.send(0);
                false
            } else {
                true
            };
        PhysicsStepToken {
            rx,
            dispatcher: Arc::clone(&self.dispatcher),
            frame_id,
            await_publish,
        }
    }
}
