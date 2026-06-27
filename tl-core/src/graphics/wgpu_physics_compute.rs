//! WGPU-backed ParadoxPE physics compute bridge.
//!
//! This is the first real CPU + secondary-iGPU physics slice. ParadoxPE remains authoritative on
//! the CPU side, while this backend offloads the integration arithmetic for large body batches to a
//! non-present integrated GPU when one is available. Every dispatch is fail-soft: if adapter
//! selection, upload, execution, or readback fails, ParadoxPE simply runs the CPU integration path.

use std::borrow::Cow;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use nalgebra::Vector3;
use paradoxpe::{
    Aabb, BodyIntegrateComputeRecord, BodyKind, BodyRegistry, PhysicsComputeBackend,
    PhysicsComputeBackendKind, PhysicsComputeDispatchRequest, PhysicsComputeDispatchResult,
    PhysicsComputeLane, PhysicsComputeStage,
};
use wgpu::{AdapterInfo, DeviceType};

use gms::safe_default_required_limits_for_adapter;

const LOCAL_SIZE_X: u32 = 128;
const DEFAULT_READBACK_TIMEOUT_MS: u64 = 250;

const INTEGRATE_WGSL: &str = r#"
struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    force: vec4<f32>,
    local_min: vec4<f32>,
    local_max: vec4<f32>,
    params: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
};

struct Params {
    gravity_dt: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> bodies: array<Body>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.counts.x) {
        return;
    }

    var body = bodies[index];
    let dt = params.gravity_dt.w;
    let gravity = params.gravity_dt.xyz;
    let kind = u32(body.params.z + 0.5);
    let is_awake = body.params.w > 0.5;

    if (kind == 0u) {
        body.force = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if (kind == 1u) {
        body.position = vec4<f32>(body.position.xyz + body.velocity.xyz * dt, body.position.w);
    } else {
        if (!is_awake) {
            body.force = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            bodies[index] = body;
            return;
        }
        let inverse_mass = body.params.x;
        let damping = clamp(body.params.y, 0.0, 0.95);
        let acceleration = gravity + body.force.xyz * inverse_mass;
        body.velocity = vec4<f32>(
            (body.velocity.xyz + acceleration * dt) * (1.0 - damping),
            body.velocity.w
        );
        body.position = vec4<f32>(body.position.xyz + body.velocity.xyz * dt, body.position.w);
        body.force = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    body.aabb_min = vec4<f32>(body.local_min.xyz + body.position.xyz, body.aabb_min.w);
    body.aabb_max = vec4<f32>(body.local_max.xyz + body.position.xyz, body.aabb_max.w);
    bodies[index] = body;
}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPhysicsComputeConfig {
    pub enable_integrate_stage: bool,
    pub min_body_count: usize,
    pub readback_timeout_ms: u64,
}

impl Default for WgpuPhysicsComputeConfig {
    fn default() -> Self {
        Self {
            enable_integrate_stage: true,
            min_body_count: 8_192,
            readback_timeout_ms: DEFAULT_READBACK_TIMEOUT_MS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPhysicsComputeCapabilities {
    pub adapter_name: String,
    pub adapter_backend: wgpu::Backend,
    pub adapter_type: DeviceType,
    pub max_storage_buffer_binding_size: u32,
    pub integrate_stage_ready: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuPhysicsDispatchPlan {
    pub stage: PhysicsComputeStage,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuIntegrateBody {
    position: [f32; 4],
    velocity: [f32; 4],
    force: [f32; 4],
    local_min: [f32; 4],
    local_max: [f32; 4],
    params: [f32; 4],
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuIntegrateBody {}
unsafe impl bytemuck::Pod for GpuIntegrateBody {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuIntegrateParams {
    gravity_dt: [f32; 4],
    counts: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuIntegrateParams {}
unsafe impl bytemuck::Pod for GpuIntegrateParams {}

#[derive(Default)]
struct WgpuPhysicsScratch {
    cpu_records: Vec<BodyIntegrateComputeRecord>,
    gpu_records: Vec<GpuIntegrateBody>,
    body_capacity: usize,
    body_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    params_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

pub struct WgpuPhysicsComputeBackend {
    config: WgpuPhysicsComputeConfig,
    capabilities: WgpuPhysicsComputeCapabilities,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    scratch: Mutex<WgpuPhysicsScratch>,
}

impl WgpuPhysicsComputeBackend {
    pub fn try_new_secondary_integrated(
        present_adapter_info: &AdapterInfo,
        config: WgpuPhysicsComputeConfig,
    ) -> Result<Option<Self>, String> {
        if !config.enable_integrate_stage {
            return Ok(None);
        }

        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let Some(adapter) = adapters.into_iter().find(|adapter| {
            let info = adapter.get_info();
            matches!(info.device_type, DeviceType::IntegratedGpu)
                && !same_physical_adapter(&info, present_adapter_info)
        }) else {
            return Ok(None);
        };
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (required_limits, _) = safe_default_required_limits_for_adapter(&adapter);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("paradoxpe-secondary-igpu-physics-device"),
            required_limits,
            ..Default::default()
        }))
        .map_err(|err| {
            format!(
                "failed to request secondary iGPU physics device '{}': {err}",
                info.name
            )
        })?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paradoxpe-wgpu-integrate-shader"),
            source: wgpu::ShaderSource::Wgsl(INTEGRATE_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paradoxpe-wgpu-integrate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paradoxpe-wgpu-integrate-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("paradoxpe-wgpu-integrate-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Some(Self {
            config,
            capabilities: WgpuPhysicsComputeCapabilities {
                adapter_name: info.name,
                adapter_backend: info.backend,
                adapter_type: info.device_type,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                integrate_stage_ready: true,
                fallback_reason: None,
            },
            device,
            queue,
            bind_group_layout,
            pipeline,
            scratch: Mutex::new(WgpuPhysicsScratch::default()),
        }))
    }

    pub fn config(&self) -> &WgpuPhysicsComputeConfig {
        &self.config
    }

    pub fn capabilities(&self) -> &WgpuPhysicsComputeCapabilities {
        &self.capabilities
    }

    pub fn build_dispatch_plan(
        &self,
        request: PhysicsComputeDispatchRequest,
    ) -> Option<WgpuPhysicsDispatchPlan> {
        if request.stage != PhysicsComputeStage::Integrate {
            return None;
        }
        if request.body_count < self.config.min_body_count {
            return None;
        }
        let workgroup_count_x =
            ((request.body_count as u32).saturating_add(LOCAL_SIZE_X - 1)) / LOCAL_SIZE_X;
        Some(WgpuPhysicsDispatchPlan {
            stage: request.stage,
            local_size_x: LOCAL_SIZE_X,
            workgroup_count_x: workgroup_count_x.max(1),
        })
    }

    fn ensure_scratch_buffers(
        &self,
        scratch: &mut WgpuPhysicsScratch,
        body_count: usize,
    ) -> Result<(), Cow<'static, str>> {
        let capacity = body_count.next_power_of_two().max(1);
        if scratch.body_capacity >= body_count
            && scratch.body_buffer.is_some()
            && scratch.readback_buffer.is_some()
            && scratch.params_buffer.is_some()
            && scratch.bind_group.is_some()
        {
            return Ok(());
        }

        let body_bytes = (capacity as u64)
            .saturating_mul(std::mem::size_of::<GpuIntegrateBody>() as u64)
            .max(16);
        let max_binding = self.capabilities.max_storage_buffer_binding_size as u64;
        if body_bytes > max_binding {
            return Err(format!(
                "secondary iGPU storage buffer limit exceeded ({} > {} bytes)",
                body_bytes, max_binding
            )
            .into());
        }

        let body_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paradoxpe-wgpu-integrate-bodies"),
            size: body_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paradoxpe-wgpu-integrate-readback"),
            size: body_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paradoxpe-wgpu-integrate-params"),
            size: std::mem::size_of::<GpuIntegrateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paradoxpe-wgpu-integrate-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: body_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        scratch.body_capacity = capacity;
        scratch.body_buffer = Some(body_buffer);
        scratch.readback_buffer = Some(readback_buffer);
        scratch.params_buffer = Some(params_buffer);
        scratch.bind_group = Some(bind_group);
        Ok(())
    }
}

impl PhysicsComputeBackend for WgpuPhysicsComputeBackend {
    fn backend_name(&self) -> &str {
        "wgpu-secondary-igpu-physics-compute"
    }

    fn backend_kind(&self) -> PhysicsComputeBackendKind {
        PhysicsComputeBackendKind::Wgpu
    }

    fn compute_lane(&self) -> PhysicsComputeLane {
        PhysicsComputeLane::CpuIgpu
    }

    fn adapter_name(&self) -> Option<&str> {
        Some(&self.capabilities.adapter_name)
    }

    fn supports_stage(&self, stage: PhysicsComputeStage) -> bool {
        matches!(stage, PhysicsComputeStage::Integrate)
            && self.config.enable_integrate_stage
            && self.capabilities.integrate_stage_ready
    }

    fn dispatch_integrate(
        &self,
        request: PhysicsComputeDispatchRequest,
        bodies: &mut BodyRegistry,
        _shards: &[std::ops::Range<usize>],
        gravity: Vector3<f32>,
        _fixed_dt: f32,
    ) -> PhysicsComputeDispatchResult {
        if !self.supports_stage(request.stage) {
            return PhysicsComputeDispatchResult::Fallback {
                reason: self
                    .capabilities
                    .fallback_reason
                    .clone()
                    .unwrap_or_else(|| "secondary iGPU integrate stage is unavailable".to_string())
                    .into(),
            };
        }
        let Some(plan) = self.build_dispatch_plan(request) else {
            return PhysicsComputeDispatchResult::Fallback {
                reason: format!(
                    "integrate body threshold not reached for secondary iGPU compute ({} < {})",
                    request.body_count, self.config.min_body_count
                )
                .into(),
            };
        };

        let started = Instant::now();
        let mut scratch = match self.scratch.lock() {
            Ok(scratch) => scratch,
            Err(_) => {
                return PhysicsComputeDispatchResult::Fallback {
                    reason: "secondary iGPU physics scratch lock poisoned".into(),
                };
            }
        };
        bodies.export_integrate_compute_records(&mut scratch.cpu_records);
        scratch.gpu_records.clear();
        let cpu_records = std::mem::take(&mut scratch.cpu_records);
        scratch
            .gpu_records
            .extend(cpu_records.iter().map(GpuIntegrateBody::from_record));
        scratch.cpu_records = cpu_records;
        if let Err(reason) = self.ensure_scratch_buffers(&mut scratch, request.body_count) {
            return PhysicsComputeDispatchResult::Fallback { reason };
        }

        let body_bytes = (scratch.gpu_records.len() as u64)
            .saturating_mul(std::mem::size_of::<GpuIntegrateBody>() as u64);
        let body_buffer = scratch
            .body_buffer
            .as_ref()
            .expect("body buffer should be allocated");
        let readback_buffer = scratch
            .readback_buffer
            .as_ref()
            .expect("readback buffer should be allocated");
        let params_buffer = scratch
            .params_buffer
            .as_ref()
            .expect("params buffer should be allocated");
        let bind_group = scratch
            .bind_group
            .as_ref()
            .expect("bind group should be allocated");
        let params = GpuIntegrateParams {
            gravity_dt: [gravity.x, gravity.y, gravity.z, request.fixed_dt],
            counts: [request.body_count as u32, 0, 0, 0],
        };
        self.queue.write_buffer(
            body_buffer,
            0,
            bytemuck::cast_slice(scratch.gpu_records.as_slice()),
        );
        self.queue
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paradoxpe-wgpu-integrate-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("paradoxpe-wgpu-integrate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(plan.workgroup_count_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(body_buffer, 0, readback_buffer, 0, body_bytes);
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        let readback_slice = readback_buffer.slice(0..body_bytes);
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        match rx.recv_timeout(Duration::from_millis(self.config.readback_timeout_ms)) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return PhysicsComputeDispatchResult::Fallback {
                    reason: format!("secondary iGPU readback map failed: {err:?}").into(),
                };
            }
            Err(_) => {
                return PhysicsComputeDispatchResult::Fallback {
                    reason: format!(
                        "secondary iGPU readback timed out after {} ms",
                        self.config.readback_timeout_ms
                    )
                    .into(),
                };
            }
        }

        let output_records = {
            let view = readback_slice.get_mapped_range();
            let output = bytemuck::cast_slice::<u8, GpuIntegrateBody>(&view);
            output
                .iter()
                .take(request.body_count)
                .map(|body| body.to_record())
                .collect::<Vec<_>>()
        };
        readback_buffer.unmap();
        if !bodies.apply_integrate_compute_records(&output_records) {
            return PhysicsComputeDispatchResult::Fallback {
                reason: "secondary iGPU output length no longer matches body registry".into(),
            };
        }

        PhysicsComputeDispatchResult::Executed {
            gpu_time_us: duration_us(started.elapsed()),
            workgroups: plan.workgroup_count_x,
        }
    }
}

impl GpuIntegrateBody {
    fn from_record(record: &BodyIntegrateComputeRecord) -> Self {
        Self {
            position: vec3_to_vec4(record.position, 0.0),
            velocity: vec3_to_vec4(record.linear_velocity, 0.0),
            force: vec3_to_vec4(record.accumulated_force, 0.0),
            local_min: vec3_to_vec4(record.local_bounds.min, 0.0),
            local_max: vec3_to_vec4(record.local_bounds.max, 0.0),
            params: [
                record.inverse_mass,
                record.linear_damping,
                body_kind_to_f32(record.kind),
                if record.awake { 1.0 } else { 0.0 },
            ],
            aabb_min: vec3_to_vec4(record.aabb.min, 0.0),
            aabb_max: vec3_to_vec4(record.aabb.max, 0.0),
        }
    }

    fn to_record(self) -> BodyIntegrateComputeRecord {
        BodyIntegrateComputeRecord {
            position: vec4_to_vec3(self.position),
            linear_velocity: vec4_to_vec3(self.velocity),
            accumulated_force: vec4_to_vec3(self.force),
            local_bounds: Aabb::new(vec4_to_vec3(self.local_min), vec4_to_vec3(self.local_max)),
            aabb: Aabb::new(vec4_to_vec3(self.aabb_min), vec4_to_vec3(self.aabb_max)),
            inverse_mass: self.params[0],
            linear_damping: self.params[1],
            kind: body_kind_from_f32(self.params[2]),
            awake: self.params[3] > 0.5,
        }
    }
}

fn same_physical_adapter(left: &AdapterInfo, right: &AdapterInfo) -> bool {
    left.vendor == right.vendor
        && left.device == right.device
        && left.backend == right.backend
        && left.name == right.name
}

fn vec3_to_vec4(v: Vector3<f32>, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

fn vec4_to_vec3(v: [f32; 4]) -> Vector3<f32> {
    Vector3::new(v[0], v[1], v[2])
}

fn body_kind_to_f32(kind: BodyKind) -> f32 {
    match kind {
        BodyKind::Static => 0.0,
        BodyKind::Kinematic => 1.0,
        BodyKind::Dynamic => 2.0,
    }
}

fn body_kind_from_f32(value: f32) -> BodyKind {
    BodyKind::from_u32(value.round().max(0.0) as u32).unwrap_or(BodyKind::Static)
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_body_record_roundtrip_preserves_integration_state() {
        let record = BodyIntegrateComputeRecord {
            position: Vector3::new(1.0, 2.0, 3.0),
            linear_velocity: Vector3::new(0.5, -0.25, 0.75),
            accumulated_force: Vector3::new(4.0, 5.0, 6.0),
            local_bounds: Aabb::from_center_half_extents(
                Vector3::zeros(),
                Vector3::new(0.25, 0.5, 0.75),
            ),
            aabb: Aabb::from_center_half_extents(
                Vector3::new(1.0, 2.0, 3.0),
                Vector3::new(0.25, 0.5, 0.75),
            ),
            inverse_mass: 0.5,
            linear_damping: 0.025,
            kind: BodyKind::Dynamic,
            awake: true,
        };

        let roundtrip = GpuIntegrateBody::from_record(&record).to_record();
        assert_eq!(roundtrip, record);
    }
}
