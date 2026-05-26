//! Async physics step runner via MPS (Multi-Processing Scaler).
//!
//! The runner now targets the bare-metal `TaskDispatcher` path so runtime
//! overlap semantics match the low-latency MPS frame pipeline. The world stays
//! behind `Arc<Mutex<_>>` for now so the rest of TLApp can keep its current
//! synchronous borrow flow while we migrate internal world phases away from the
//! remaining single-step hot path.

use crate::PerformanceProfile;
use mps::{
    DispatcherPhaseCallbacks, DispatcherPhasePlan, MpsCompressionConfig, MpsPerformanceProfile,
    MpsThreadPoolMetrics, MpsTuningProfile, PhysicsDispatchTrigger, TaskDispatcher,
    TaskDispatcherConfig,
};
use paradoxpe::{parallel::JobQueue, PhysicsWorld};

struct MpsJobQueue {
    dispatcher: Arc<TaskDispatcher>,
}

impl JobQueue for MpsJobQueue {
    fn submit(&self, f: Box<dyn FnOnce() + Send>) {
        self.dispatcher.submit_paradox_job(f);
    }

    fn try_execute_one(&self) -> bool {
        self.dispatcher.try_execute_paradox_job()
    }
}
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::Duration;
use tl_core::write_world_render_transforms_to_dispatcher_storage;

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
}

impl PhysicsMpsRunner {
    /// Create a runner from an existing world. Spawns the MPS worker pool.
    pub fn new(world: PhysicsWorld, profile: PerformanceProfile) -> Self {
        let mut dispatcher_config = TaskDispatcherConfig::default();
        let mps_profile = match profile {
            PerformanceProfile::Balanced => MpsPerformanceProfile::Balanced,
            PerformanceProfile::Aggressive => MpsPerformanceProfile::Aggressive,
            PerformanceProfile::Heimdall => MpsPerformanceProfile::Heimdall,
        };
        let tuning = MpsTuningProfile::from_profile(mps_profile);
        dispatcher_config.apply_tuning(&tuning);
        dispatcher_config.compression = MpsCompressionConfig::from_tileline_env();
        dispatcher_config.queue_capacity = dispatcher_config.queue_capacity.max(262_144);
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
        paradoxpe::parallel::set_global_job_queue(Arc::new(MpsJobQueue {
            dispatcher: Arc::clone(&dispatcher),
        }));
        Self {
            world: Arc::new(Mutex::new(world)),
            dispatcher,
            next_frame_id: AtomicU64::new(1),
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
            compression: metrics.compression,
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

        let _substeps = plan.substeps;
        let plan = Arc::new(plan);

        let integrate_world = Arc::clone(&world);
        let integrate_plan = Arc::clone(&plan);
        let integrate = Arc::new(move |_ctx: &mps::DispatcherTaskContext| {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let mut world = integrate_world.lock().unwrap();
                let mut timings = paradoxpe::PhysicsStepTimings::default();
                for step_index in 0..integrate_plan.substeps {
                    world.execute_integrate_phase(&integrate_plan, step_index, &mut timings);
                }
                world.last_step_timings.integrate_us = timings.integrate_us;
                world.last_step_timings.integrate_mode = timings.integrate_mode;
                world.last_step_timings.integrate_serial_fallback_reason =
                    timings.integrate_serial_fallback_reason;
            }));
        });

        let broadphase_world = Arc::clone(&world);
        let broadphase_plan = Arc::clone(&plan);
        let broadphase = Arc::new(move |_ctx: &mps::DispatcherTaskContext| {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let mut world = broadphase_world.lock().unwrap();
                let mut timings = paradoxpe::PhysicsStepTimings::default();
                for _ in 0..broadphase_plan.substeps {
                    world.execute_broadphase_phase(&broadphase_plan, &mut timings);
                }
                world.last_step_timings.broadphase_us = timings.broadphase_us;
                world.last_step_timings.broadphase_mode = timings.broadphase_mode;
                world.last_step_timings.broadphase_serial_fallback_reason =
                    timings.broadphase_serial_fallback_reason;
                world.last_step_timings.candidate_pairs = timings.candidate_pairs;
            }));
        });

        let narrowphase_world = Arc::clone(&world);
        let narrowphase_plan = Arc::clone(&plan);
        let narrowphase = Arc::new(move |_ctx: &mps::DispatcherTaskContext| {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let mut world = narrowphase_world.lock().unwrap();
                let mut timings = paradoxpe::PhysicsStepTimings::default();
                for _ in 0..narrowphase_plan.substeps {
                    world.execute_narrowphase_and_solver_phase(&narrowphase_plan, &mut timings);
                }
                world.last_step_timings.narrowphase_us = timings.narrowphase_us;
                world.last_step_timings.narrowphase_mode = timings.narrowphase_mode;
                world.last_step_timings.narrowphase_serial_fallback_reason =
                    timings.narrowphase_serial_fallback_reason;
                world.last_step_timings.solver_us = timings.solver_us;
                world.last_step_timings.solver_mode = timings.solver_mode;
                world.last_step_timings.solver_serial_fallback_reason =
                    timings.solver_serial_fallback_reason;
                world.last_step_timings.manifold_count = timings.manifold_count;
            }));
        });

        let sleep_world = Arc::clone(&world);
        let sleep_plan = Arc::clone(&plan);
        let sleep_tx = tx;
        let sleep_finalize = Arc::new(move |ctx: &mps::DispatcherTaskContext| {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut world = sleep_world.lock().unwrap();
                let mut timings = paradoxpe::PhysicsStepTimings::default();
                for _ in 0..sleep_plan.substeps {
                    world.execute_sleep_phase(&sleep_plan, &mut timings);
                }
                world.last_step_timings.sleep_us = timings.sleep_us;
                world.last_step_timings.substeps = sleep_plan.substeps as u32;
                write_world_render_transforms_to_dispatcher_storage(
                    &world,
                    ctx.transforms.as_ref(),
                    ctx.physics_write_slot,
                );
                sleep_plan.substeps
            }));
            match result {
                Ok(completed) => {
                    let _ = sleep_tx.send(completed);
                }
                Err(_) => {
                    eprintln!(
                        "[physics mps] frame {} sleep phase panicked; returning 0 substeps",
                        ctx.frame_id
                    );
                    let _ = sleep_tx.send(0);
                }
            }
        });

        let callbacks = DispatcherPhaseCallbacks::default()
            .with_broadphase(broadphase)
            .with_narrowphase(narrowphase)
            .with_integration(integrate)
            .with_sleep_finalize(sleep_finalize);
        let trigger = PhysicsDispatchTrigger::with_phase_plans(
            frame_id,
            DispatcherPhasePlan::new(1, 1),
            DispatcherPhasePlan::new(1, 1),
            DispatcherPhasePlan::new(1, 1),
            DispatcherPhasePlan::new(1, 1),
            DispatcherPhasePlan::new(1, 1),
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
