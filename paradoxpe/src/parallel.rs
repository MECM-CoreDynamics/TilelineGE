//! Lightweight parallel helpers for ParadoxPE hot paths.
//!
//! When a global JobQueue is installed (e.g. MPS paradox_queue), helpers
//! submit chunk work to that queue instead of spawning scoped OS threads.
//! The queue must outlive any scope; chunk pointers are valid until the
//! counter drops to zero.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

/// Execution mode snapshot returned by ParadoxPE parallel helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelExecutionMode {
    /// Helper path was not executed for this phase.
    #[default]
    NotRun,
    /// Work was processed in parallel chunks.
    Parallel,
    /// Sequential fallback because only one logical worker is available.
    SerialSingleWorker,
    /// Sequential fallback because workload is below parallel threshold.
    SerialSmallWorkload,
    /// Sequential fallback because input shards are not parallel-safe.
    SerialUnsupportedPlan,
    /// Sequential fallback because parallel execution is not implemented yet.
    SerialUnimplemented,
}

impl ParallelExecutionMode {
    #[inline]
    pub const fn is_parallel(self) -> bool {
        matches!(self, Self::Parallel)
    }

    #[inline]
    pub const fn is_serial(self) -> bool {
        matches!(
            self,
            Self::SerialSingleWorker
                | Self::SerialSmallWorkload
                | Self::SerialUnsupportedPlan
                | Self::SerialUnimplemented
        )
    }

    #[inline]
    pub const fn serial_fallback_reason(self) -> Option<&'static str> {
        match self {
            Self::SerialSingleWorker => Some("single_worker"),
            Self::SerialSmallWorkload => Some("small_workload"),
            Self::SerialUnsupportedPlan => Some("unsupported_plan"),
            Self::SerialUnimplemented => Some("parallel_not_implemented"),
            _ => None,
        }
    }
}

thread_local! {
    static EXTERNAL_PARALLEL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Guard that suppresses ParadoxPE's internal parallel helpers while an
/// external dispatcher owns the frame.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ExternalParallelGuard;

impl Drop for ExternalParallelGuard {
    fn drop(&mut self) {
        EXTERNAL_PARALLEL_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

#[allow(dead_code)]
pub(crate) fn enter_external_parallel_mode() -> ExternalParallelGuard {
    EXTERNAL_PARALLEL_DEPTH.with(|depth| {
        depth.set(depth.get().saturating_add(1));
    });
    ExternalParallelGuard
}

#[inline]
fn external_parallel_mode_active() -> bool {
    EXTERNAL_PARALLEL_DEPTH.with(|depth| depth.get() > 0)
}

/// Return the logical worker count available to this process.
#[inline]
pub fn worker_count() -> usize {
    if external_parallel_mode_active() {
        return 1;
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

#[inline]
fn resolve_execution_mode(
    total_items: usize,
    min_items_per_worker: usize,
) -> ParallelExecutionMode {
    let workers = worker_count();
    if workers <= 1 {
        return ParallelExecutionMode::SerialSingleWorker;
    }
    if total_items < min_items_per_worker.max(1).saturating_mul(2) {
        return ParallelExecutionMode::SerialSmallWorkload;
    }
    ParallelExecutionMode::Parallel
}

#[inline]
fn chunk_len(total_items: usize, min_items_per_worker: usize) -> usize {
    let workers = worker_count().max(1);
    total_items
        .div_ceil(workers)
        .max(min_items_per_worker.max(1))
        .max(1)
}

/// External job queue abstraction used by ParadoxPE parallel helpers.
pub trait JobQueue: Send + Sync {
    /// Submit a chunk of work to the queue.
    fn submit(&self, f: Box<dyn FnOnce() + Send>);

    /// Try to execute one pending job on the caller thread.
    /// Returns true if a job was executed.
    fn try_execute_one(&self) -> bool;
}

static GLOBAL_JOB_QUEUE: OnceLock<Arc<dyn JobQueue + Send + Sync>> = OnceLock::new();

/// Install a global job queue for ParadoxPE parallel helpers.
pub fn set_global_job_queue(queue: Arc<dyn JobQueue + Send + Sync>) {
    let _ = GLOBAL_JOB_QUEUE.set(queue);
}

fn global_job_queue() -> Option<&'static Arc<dyn JobQueue + Send + Sync>> {
    GLOBAL_JOB_QUEUE.get()
}

/// Wait until all submitted jobs finish, helping execute along the way.
fn wait_and_help(counter: &Arc<AtomicUsize>) {
    while counter.load(Ordering::Acquire) > 0 {
        if let Some(queue) = global_job_queue() {
            if queue.try_execute_one() {
                counter.fetch_sub(1, Ordering::Release);
            } else {
                std::hint::spin_loop();
            }
        } else {
            break;
        }
    }
}

/// Parallel-for over a mutable slice with stable global index.
pub fn for_each_mut_indexed<T, F>(
    slice: &mut [T],
    min_items_per_worker: usize,
    f: F,
) -> ParallelExecutionMode
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    let mode = resolve_execution_mode(slice.len(), min_items_per_worker);
    if !mode.is_parallel() {
        for (index, item) in slice.iter_mut().enumerate() {
            f(index, item);
        }
        return mode;
    }

    let chunk = chunk_len(slice.len(), min_items_per_worker);

    if let Some(queue) = global_job_queue() {
        let counter = Arc::new(AtomicUsize::new(0));
        // SAFETY: f outlives all submitted jobs because we block on counter.
        let f_addr = &f as *const F as usize;
        for (chunk_index, chunk_slice) in slice.chunks_mut(chunk).enumerate() {
            let base = chunk_index * chunk;
            let ptr = chunk_slice.as_mut_ptr() as usize;
            let len = chunk_slice.len();
            counter.fetch_add(1, Ordering::Release);
            queue.submit(Box::new(move || {
                let f_ref = unsafe { &*(f_addr as *const F) };
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) };
                for (offset, item) in slice.iter_mut().enumerate() {
                    f_ref(base + offset, item);
                }
            }));
        }
        wait_and_help(&counter);
        return ParallelExecutionMode::Parallel;
    }

    thread::scope(|scope| {
        for (chunk_index, chunk_slice) in slice.chunks_mut(chunk).enumerate() {
            let base = chunk_index * chunk;
            let f_ref = &f;
            scope.spawn(move || {
                for (offset, item) in chunk_slice.iter_mut().enumerate() {
                    f_ref(base + offset, item);
                }
            });
        }
    });
    ParallelExecutionMode::Parallel
}

/// Parallel-for over `0..len` with deterministic chunk order.
pub fn for_each_index<F>(len: usize, min_items_per_worker: usize, f: F) -> ParallelExecutionMode
where
    F: Fn(usize) + Sync,
{
    let mode = resolve_execution_mode(len, min_items_per_worker);
    if !mode.is_parallel() {
        for index in 0..len {
            f(index);
        }
        return mode;
    }

    let chunk = chunk_len(len, min_items_per_worker);

    if let Some(queue) = global_job_queue() {
        let counter = Arc::new(AtomicUsize::new(0));
        // SAFETY: f outlives all submitted jobs because we block on counter.
        let f_addr = &f as *const F as usize;
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            counter.fetch_add(1, Ordering::Release);
            queue.submit(Box::new(move || {
                let f_ref = unsafe { &*(f_addr as *const F) };
                for index in start..end {
                    f_ref(index);
                }
            }));
        }
        wait_and_help(&counter);
        return ParallelExecutionMode::Parallel;
    }

    thread::scope(|scope| {
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            let f_ref = &f;
            scope.spawn(move || {
                for index in start..end {
                    f_ref(index);
                }
            });
        }
    });
    ParallelExecutionMode::Parallel
}

/// Collect values with a parallel filter-map over a read-only slice.
///
/// Output order is deterministic and follows input order by chunk.
pub fn collect_filter_map<T, U, F>(
    input: &[T],
    min_items_per_worker: usize,
    map: F,
) -> (Vec<U>, ParallelExecutionMode)
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync,
{
    let mode = resolve_execution_mode(input.len(), min_items_per_worker);
    if !mode.is_parallel() {
        let mut out = Vec::with_capacity(input.len() / 2);
        for item in input {
            if let Some(mapped) = map(item) {
                out.push(mapped);
            }
        }
        return (out, mode);
    }

    let chunk = chunk_len(input.len(), min_items_per_worker);



    let mut chunk_outputs: Vec<Vec<U>> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk_slice in input.chunks(chunk) {
            let map_ref = &map;
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(chunk_slice.len() / 2);
                for item in chunk_slice {
                    if let Some(mapped) = map_ref(item) {
                        local.push(mapped);
                    }
                }
                local
            }));
        }
        chunk_outputs.reserve(handles.len());
        for handle in handles {
            chunk_outputs.push(
                handle
                    .join()
                    .expect("ParadoxPE parallel collect worker panicked"),
            );
        }
    });

    let total = chunk_outputs.iter().map(Vec::len).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    for mut local in chunk_outputs {
        out.append(&mut local);
    }
    (out, ParallelExecutionMode::Parallel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn for_each_mut_indexed_visits_every_index_once() {
        let mut data = vec![0usize; 4096];
        let mode = for_each_mut_indexed(&mut data, 64, |index, item| {
            *item = index + 1;
        });
        assert!(mode.is_parallel() || mode.is_serial());
        for (index, value) in data.iter().enumerate() {
            assert_eq!(*value, index + 1);
        }
    }

    #[test]
    fn collect_filter_map_keeps_chunk_deterministic_order() {
        let data = (0u32..2048).collect::<Vec<_>>();
        let (out, mode) = collect_filter_map(
            &data,
            64,
            |value| {
                if value % 3 == 0 {
                    Some(*value)
                } else {
                    None
                }
            },
        );
        assert!(mode.is_parallel() || mode.is_serial());
        let expected = data
            .iter()
            .copied()
            .filter(|value| value % 3 == 0)
            .collect::<Vec<_>>();
        assert_eq!(out, expected);
    }

    #[test]
    fn collect_filter_map_runs_map_on_all_items() {
        let data = (0u32..1024).collect::<Vec<_>>();
        let calls = AtomicUsize::new(0);
        let (_out, mode) = collect_filter_map(&data, 32, |value| {
            calls.fetch_add(1, Ordering::Relaxed);
            Some(*value)
        });
        assert!(mode.is_parallel() || mode.is_serial());
        assert_eq!(calls.load(Ordering::Relaxed), data.len());
    }

    #[test]
    fn serial_fallback_reason_is_present_for_all_serial_variants() {
        assert_eq!(
            ParallelExecutionMode::SerialSingleWorker.serial_fallback_reason(),
            Some("single_worker")
        );
        assert_eq!(
            ParallelExecutionMode::SerialSmallWorkload.serial_fallback_reason(),
            Some("small_workload")
        );
        assert_eq!(
            ParallelExecutionMode::SerialUnsupportedPlan.serial_fallback_reason(),
            Some("unsupported_plan")
        );
        assert_eq!(
            ParallelExecutionMode::SerialUnimplemented.serial_fallback_reason(),
            Some("parallel_not_implemented")
        );
        assert!(ParallelExecutionMode::Parallel.serial_fallback_reason().is_none());
        assert!(ParallelExecutionMode::NotRun.serial_fallback_reason().is_none());
    }

    #[test]
    fn for_each_index_reports_serial_for_workload_below_threshold() {
        let calls = AtomicUsize::new(0);
        let mode = for_each_index(3, 64, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            mode.is_serial(),
            "tiny workload must fall back to serial, got {mode:?}"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 3, "all items must still be visited");
    }

    #[test]
    fn for_each_mut_indexed_reports_serial_for_empty_slice() {
        let mut data: Vec<u32> = Vec::new();
        let mode = for_each_mut_indexed(&mut data, 16, |_, _| {});
        assert!(
            mode.is_serial(),
            "empty slice must fall back to serial, got {mode:?}"
        );
    }

    #[test]
    fn for_each_index_reports_parallel_for_large_workload_on_multi_worker_host() {
        if worker_count() <= 1 {
            return;
        }
        let calls = AtomicUsize::new(0);
        let mode = for_each_index(4096, 64, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(
            mode,
            ParallelExecutionMode::Parallel,
            "large workload on multi-worker host must run parallel"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 4096);
    }

    #[test]
    fn for_each_mut_indexed_reports_parallel_for_large_workload_on_multi_worker_host() {
        if worker_count() <= 1 {
            return;
        }
        let mut data = vec![0u64; 4096];
        let mode = for_each_mut_indexed(&mut data, 64, |i, v| {
            *v = i as u64 + 1;
        });
        assert_eq!(
            mode,
            ParallelExecutionMode::Parallel,
            "large workload on multi-worker host must run parallel"
        );
        for (i, v) in data.iter().enumerate() {
            assert_eq!(*v, i as u64 + 1);
        }
    }
}
