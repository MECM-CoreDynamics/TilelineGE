//! Runtime-owned unified bridge orchestration for GMS/MGS paths.
//!
//! This module provides one frame-pipeline surface over two bridge backends:
//! - `FrameLoopRuntime` (MPS<->GMS)
//! - `MgsFrameLoopRuntime` (MPS<->MGS)
//!
//! It enables TLApp to keep one overlapped frame loop while selecting GPU planner
//! path by scheduler policy.

use gms::SceneWorkloadEstimate;
use mgs::MobileGpuProfile;
use tl_core::{
    BridgeFramePlan, MgsBridgeFramePlan, MlsBackendKind, MlsExecutionMode, MlsPrecisionMode,
    MlsRuntime, MlsRuntimeConfig, MlsTelemetry, MlsWorkloadKind, MpsGmsBridgeConfig,
    MpsMgsBridgeConfig,
};

use crate::frame_loop::{FrameLoopRuntime, FrameLoopRuntimeConfig};
use crate::mgs_frame_loop::{MgsFrameLoopRuntime, MgsFrameLoopRuntimeConfig};
use crate::mobile_scene_dispatch::{
    submit_mobile_hint_to_bridge, MobileSceneDispatchConfig, MobileSceneDispatchSubmission,
};
use crate::mobile_scene_workload::{
    estimate_mobile_workload_hint, MobileSceneWorkloadBridgeConfig,
};
use crate::scene::SceneFrameInstances;
use crate::scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchSubmission,
};
use crate::scene_workload::{estimate_scene_workload_requests, SceneWorkloadBridgeConfig};
use crate::GraphicsSchedulerPath;

/// Active runtime bridge path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBridgePath {
    /// Desktop/full multi-GPU planning path.
    GmsPath,
    /// Mobile/TBDR planning path.
    MgsPath,
}

impl RuntimeBridgePath {
    /// Human-readable short label for telemetry overlays.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GmsPath => "gms",
            Self::MgsPath => "mgs",
        }
    }
}

/// Unified frame plan wrapper returned by runtime bridge orchestration.
#[derive(Debug, Clone)]
pub enum RuntimeFramePlan {
    Gms(BridgeFramePlan),
    Mgs(MgsBridgeFramePlan),
}

impl RuntimeFramePlan {
    /// Frame id attached to the wrapped bridge plan.
    pub fn frame_id(&self) -> u64 {
        match self {
            Self::Gms(plan) => plan.frame_id,
            Self::Mgs(plan) => plan.frame_id,
        }
    }

    /// Source path of this plan.
    pub fn path(&self) -> RuntimeBridgePath {
        match self {
            Self::Gms(_) => RuntimeBridgePath::GmsPath,
            Self::Mgs(_) => RuntimeBridgePath::MgsPath,
        }
    }
}

/// Unified scene submission result.
#[derive(Debug, Clone)]
pub enum RuntimeBridgeSubmission {
    Gms(SceneDispatchSubmission),
    Mgs(MobileSceneDispatchSubmission),
}

impl RuntimeBridgeSubmission {
    /// Frame id associated with this submission.
    pub fn frame_id(&self) -> u64 {
        match self {
            Self::Gms(s) => s.frame_id,
            Self::Mgs(s) => s.frame_id,
        }
    }

    /// Number of submitted CPU placeholder tasks.
    pub fn submitted_tasks(&self) -> u32 {
        match self {
            Self::Gms(s) => s.total_submitted_tasks,
            Self::Mgs(s) => s.total_submitted_tasks,
        }
    }

    /// Whether frame was sealed in this dispatch.
    pub fn frame_sealed(&self) -> bool {
        match self {
            Self::Gms(s) => s.frame_sealed,
            Self::Mgs(s) => s.frame_sealed,
        }
    }
}

/// Runtime bridge tick result with fallback diagnostics.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeTick {
    pub bridge_path: RuntimeBridgePath,
    pub bridge_pump_published: usize,
    pub bridge_pump_drained: usize,
    pub queued_plan_depth: usize,
    pub used_fallback_plan: bool,
    pub plan: Option<RuntimeFramePlan>,
}

impl RuntimeBridgeTick {
    fn empty(path: RuntimeBridgePath) -> Self {
        Self {
            bridge_path: path,
            bridge_pump_published: 0,
            bridge_pump_drained: 0,
            queued_plan_depth: 0,
            used_fallback_plan: false,
            plan: None,
        }
    }
}

/// Path-independent bridge metrics used by runtime telemetry.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeMetrics {
    pub bridge_path: RuntimeBridgePath,
    pub queued_plan_depth: usize,
    pub bridge_tick_published_frames: u64,
    pub bridge_tick_drained_plans: u64,
    pub frame_plans_popped: u64,
    pub gms_mode: Option<GmsScalerMode>,
    pub domain_budgets: Option<GmsDomainBudgets>,
    pub performance_profile: PerformanceProfile,
    pub sm_cu_utilization: f32,
    pub lane_queue_depth: usize,
    pub ai_ml_drop_rate: f32,
    pub fallback_reason: Option<String>,
    pub mls_backend: MlsBackendKind,
    pub mls_device: String,
    pub mls_active_workloads: Vec<MlsWorkloadKind>,
    pub mls_infer_queue_depth: usize,
    pub mls_train_queue_depth: usize,
    pub mls_drop_rate: f32,
    pub mls_fallback_reason: Option<String>,
    pub mls_precision: MlsPrecisionMode,
    pub mls_step_time_ms: f32,
}

/// Runtime GMS scaler mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmsScalerMode {
    Adaptive,
    Fixed,
}

impl GmsScalerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Fixed => "fixed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adaptive" | "auto" => Some(Self::Adaptive),
            "fixed" => Some(Self::Fixed),
            _ => None,
        }
    }
}

/// Cross-stack performance profile used by runtime to control MPS, GMS, and MGS aggressiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerformanceProfile {
    #[default]
    Balanced,
    Aggressive,
    Heimdall,
}

impl PerformanceProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
            Self::Heimdall => "heimdall",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "balanced" | "default" => Some(Self::Balanced),
            "aggressive" | "max" => Some(Self::Aggressive),
            "heimdall" => Some(Self::Heimdall),
            _ => None,
        }
    }
}

/// Runtime GMS guardrail profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmsGuardrailProfile {
    Balanced,
    Aggressive,
    Relaxed,
}

impl GmsGuardrailProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
            Self::Relaxed => "relaxed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "balanced" | "default" => Some(Self::Balanced),
            "aggressive" | "max" => Some(Self::Aggressive),
            "relaxed" | "soft" => Some(Self::Relaxed),
            _ => None,
        }
    }
}

/// GMS domain names used by scaler controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmsScalerDomain {
    Render,
    Physics,
    AiMl,
    PostFx,
    Ui,
}

impl GmsScalerDomain {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "render" => Some(Self::Render),
            "physics" => Some(Self::Physics),
            "ai_ml" | "aiml" | "ml" | "ai" => Some(Self::AiMl),
            "postfx" | "post_fx" | "post" => Some(Self::PostFx),
            "ui" => Some(Self::Ui),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Physics => "physics",
            Self::AiMl => "ai_ml",
            Self::PostFx => "postfx",
            Self::Ui => "ui",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GmsDomainBudgets {
    pub render_budget_pct: u8,
    pub physics_budget_pct: u8,
    pub ai_ml_budget_pct: u8,
    pub postfx_budget_pct: u8,
    pub ui_budget_pct: u8,
    /// Optional per-domain GPU pinning. `None` means follow auto-routing.
    pub render_gpu: Option<u8>,
    pub physics_gpu: Option<u8>,
    pub ai_ml_gpu: Option<u8>,
    pub postfx_gpu: Option<u8>,
    pub ui_gpu: Option<u8>,
}

impl Default for GmsDomainBudgets {
    fn default() -> Self {
        Self {
            render_budget_pct: 35,
            physics_budget_pct: 35,
            ai_ml_budget_pct: 20,
            postfx_budget_pct: 10,
            ui_budget_pct: 0,
            render_gpu: None,
            physics_gpu: None,
            ai_ml_gpu: None,
            postfx_gpu: None,
            ui_gpu: None,
        }
    }
}

/// Runtime GMS scaler configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GmsScalerConfig {
    pub mode: GmsScalerMode,
    pub target_fps: u32,
    pub min_physics_budget_pct: u8,
    pub budgets: GmsDomainBudgets,
    pub guardrail: GmsGuardrailProfile,
    pub profile: PerformanceProfile,
    /// When `true` (default) the scaler chooses GPU routing per domain automatically.
    /// When `false` the `*_gpu` fields in `budgets` are respected.
    pub auto_gpu_routing: bool,
}

impl Default for GmsScalerConfig {
    fn default() -> Self {
        Self {
            mode: GmsScalerMode::Adaptive,
            target_fps: 60,
            min_physics_budget_pct: 35,
            budgets: GmsDomainBudgets::default(),
            guardrail: GmsGuardrailProfile::Balanced,
            profile: PerformanceProfile::Balanced,
            auto_gpu_routing: true,
        }
    }
}

impl GmsScalerConfig {
    pub fn set_budget(&mut self, domain: GmsScalerDomain, pct: u8) {
        match domain {
            GmsScalerDomain::Render => self.budgets.render_budget_pct = pct.min(100),
            GmsScalerDomain::Physics => self.budgets.physics_budget_pct = pct.min(100),
            GmsScalerDomain::AiMl => self.budgets.ai_ml_budget_pct = pct.min(100),
            GmsScalerDomain::PostFx => self.budgets.postfx_budget_pct = pct.min(100),
            GmsScalerDomain::Ui => self.budgets.ui_budget_pct = pct.min(100),
        }
    }

    pub fn set_gpu(&mut self, domain: GmsScalerDomain, gpu: Option<u8>) {
        match domain {
            GmsScalerDomain::Render => self.budgets.render_gpu = gpu,
            GmsScalerDomain::Physics => self.budgets.physics_gpu = gpu,
            GmsScalerDomain::AiMl => self.budgets.ai_ml_gpu = gpu,
            GmsScalerDomain::PostFx => self.budgets.postfx_gpu = gpu,
            GmsScalerDomain::Ui => self.budgets.ui_gpu = gpu,
        }
    }
}

/// Runtime bridge configuration.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeConfig {
    pub gms_frame_loop: FrameLoopRuntimeConfig,
    pub gms_bridge: MpsGmsBridgeConfig,
    pub gms_scene_workload: SceneWorkloadBridgeConfig,
    pub gms_scene_dispatch: SceneDispatchBridgeConfig,
    pub gms_scaler: GmsScalerConfig,
    pub mls: MlsRuntimeConfig,
    pub mgs_frame_loop: MgsFrameLoopRuntimeConfig,
    pub mgs_bridge: MpsMgsBridgeConfig,
    pub mgs_scene_workload: MobileSceneWorkloadBridgeConfig,
    pub mgs_scene_dispatch: MobileSceneDispatchConfig,
    /// Reuse last valid plan when no new bridge plan is ready for current frame.
    pub allow_fallback_reuse: bool,
}

impl Default for RuntimeBridgeConfig {
    fn default() -> Self {
        Self {
            gms_frame_loop: FrameLoopRuntimeConfig::default(),
            gms_bridge: MpsGmsBridgeConfig::default(),
            gms_scene_workload: SceneWorkloadBridgeConfig::default(),
            gms_scene_dispatch: SceneDispatchBridgeConfig::default(),
            gms_scaler: GmsScalerConfig::default(),
            mls: MlsRuntimeConfig::default(),
            mgs_frame_loop: MgsFrameLoopRuntimeConfig::default(),
            mgs_bridge: MpsMgsBridgeConfig::default(),
            mgs_scene_workload: MobileSceneWorkloadBridgeConfig::default(),
            mgs_scene_dispatch: MobileSceneDispatchConfig::default(),
            allow_fallback_reuse: true,
        }
    }
}

enum RuntimeBridgeLoop {
    Gms(FrameLoopRuntime),
    Mgs(MgsFrameLoopRuntime),
}

/// Runtime-owned unified bridge orchestrator for parallel frame pipeline.
pub struct RuntimeBridgeOrchestrator {
    config: RuntimeBridgeConfig,
    path: RuntimeBridgePath,
    inner: RuntimeBridgeLoop,
    last_plan: Option<RuntimeFramePlan>,
    gms_scaler: GmsScalerConfig,
    mls: MlsRuntime,
    gms_complexity_ema: f64,
    gms_ai_ml_requested_jobs: u64,
    gms_ai_ml_kept_jobs: u64,
    gms_last_sm_cu_utilization: f32,
    gms_last_fallback_reason: Option<String>,
    gms_last_lane_queue_depth: usize,
}

impl RuntimeBridgeOrchestrator {
    /// Build orchestrator from selected scheduler path.
    pub fn new_for_scheduler(
        scheduler: GraphicsSchedulerPath,
        adapter_name: &str,
        mut config: RuntimeBridgeConfig,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        config.mgs_bridge.default_target_width = viewport_width.max(1);
        config.mgs_bridge.default_target_height = viewport_height.max(1);

        match runtime_bridge_path_from_scheduler(scheduler) {
            RuntimeBridgePath::GmsPath => Self {
                path: RuntimeBridgePath::GmsPath,
                inner: RuntimeBridgeLoop::Gms(FrameLoopRuntime::new(
                    config.gms_frame_loop,
                    config.gms_bridge,
                )),
                gms_scaler: config.gms_scaler,
                mls: MlsRuntime::new(config.mls.clone()),
                config,
                last_plan: None,
                gms_complexity_ema: 0.0,
                gms_ai_ml_requested_jobs: 0,
                gms_ai_ml_kept_jobs: 0,
                gms_last_sm_cu_utilization: 1.0,
                gms_last_fallback_reason: None,
                gms_last_lane_queue_depth: 0,
            },
            RuntimeBridgePath::MgsPath => {
                let profile = MobileGpuProfile::detect(adapter_name);
                Self {
                    path: RuntimeBridgePath::MgsPath,
                    inner: RuntimeBridgeLoop::Mgs(MgsFrameLoopRuntime::new(
                        config.mgs_frame_loop,
                        config.mgs_bridge.clone(),
                        profile,
                    )),
                    gms_scaler: config.gms_scaler,
                    mls: MlsRuntime::new(config.mls.clone()),
                    config,
                    last_plan: None,
                    gms_complexity_ema: 0.0,
                    gms_ai_ml_requested_jobs: 0,
                    gms_ai_ml_kept_jobs: 0,
                    gms_last_sm_cu_utilization: 1.0,
                    gms_last_fallback_reason: None,
                    gms_last_lane_queue_depth: 0,
                }
            }
        }
    }

    pub fn gms_scaler_config(&self) -> GmsScalerConfig {
        self.gms_scaler
    }

    pub fn mls_config(&self) -> MlsRuntimeConfig {
        self.mls.config().clone()
    }

    pub fn mls_telemetry(&self) -> MlsTelemetry {
        self.mls.telemetry().clone()
    }

    pub fn set_gms_mode(&mut self, mode: GmsScalerMode) {
        self.gms_scaler.mode = mode;
    }

    pub fn set_gms_target_fps(&mut self, target_fps: u32) {
        self.gms_scaler.target_fps = target_fps.clamp(24, 480);
    }

    pub fn set_gms_budget(&mut self, domain: GmsScalerDomain, pct: u8) {
        self.gms_scaler.set_budget(domain, pct);
    }

    pub fn set_gms_gpu(&mut self, domain: GmsScalerDomain, gpu: Option<u8>) {
        self.gms_scaler.set_gpu(domain, gpu);
    }

    pub fn set_gms_auto_gpu_routing(&mut self, enabled: bool) {
        self.gms_scaler.auto_gpu_routing = enabled;
    }

    pub fn set_gms_guardrail(&mut self, profile: GmsGuardrailProfile) {
        self.gms_scaler.guardrail = profile;
    }

    pub fn set_gms_profile(&mut self, profile: PerformanceProfile) {
        self.gms_scaler.profile = profile;
    }

    pub fn set_mls_mode(&mut self, mode: MlsExecutionMode) {
        self.mls.set_mode(mode);
    }

    pub fn set_mls_backend(&mut self, backend: MlsBackendKind) {
        self.mls.set_backend(backend);
    }

    pub fn set_mls_precision(&mut self, precision: MlsPrecisionMode) {
        self.mls.set_precision(precision);
    }

    pub fn set_mls_workload_enabled(&mut self, workload: MlsWorkloadKind, enabled: bool) {
        self.mls.set_workload_enabled(workload, enabled);
    }

    pub fn set_mls_budget(&mut self, workload: MlsWorkloadKind, pct: u8) {
        self.mls.set_budget(workload, pct);
    }

    pub fn bind_mls_model(
        &mut self,
        slot: u32,
        source: impl Into<String>,
    ) -> Result<String, String> {
        self.mls.bind_model(slot, source)
    }

    pub fn run_mls_slot(&mut self, slot: u32) -> Result<String, String> {
        self.mls.run(slot)
    }

    pub fn train_mls_slot(&mut self, slot: u32, steps: u32) -> Result<String, String> {
        self.mls.train_step(slot, steps)
    }

    pub fn checkpoint_mls_slot(
        &mut self,
        slot: u32,
        save: bool,
        name: &str,
    ) -> Result<String, String> {
        self.mls.checkpoint(slot, save, name)
    }

    pub fn gms_metric(&self, name: &str) -> Option<f64> {
        let name = name.trim().to_ascii_lowercase();
        match name.as_str() {
            "sm_cu_utilization" | "utilization" => Some(self.gms_last_sm_cu_utilization as f64),
            "lane_queue_depth" | "queue_depth" => Some(self.gms_last_lane_queue_depth as f64),
            "ai_ml_drop_rate" | "aiml_drop_rate" => {
                let requested = self.gms_ai_ml_requested_jobs.max(1);
                Some(1.0 - (self.gms_ai_ml_kept_jobs as f64 / requested as f64).clamp(0.0, 1.0))
            }
            "target_fps" => Some(self.gms_scaler.target_fps as f64),
            _ => None,
        }
    }

    pub fn mls_metric(&self, name: &str) -> Option<f64> {
        self.mls.metric(name)
    }

    pub fn gms_status_line(&self) -> Option<String> {
        if self.path != RuntimeBridgePath::GmsPath {
            return None;
        }
        let requested = self.gms_ai_ml_requested_jobs.max(1);
        let ai_ml_drop_rate =
            1.0 - (self.gms_ai_ml_kept_jobs as f64 / requested as f64).clamp(0.0, 1.0);
        let gpu_str = |gpu: Option<u8>| gpu.map(|g| format!("@GPU{g}")).unwrap_or_else(|| "@auto".to_string());
        Some(format!(
            "gms scaler | mode={} target_fps={} guardrail={} profile={} auto_gpu={} budgets[render={}{} physics={}{} ai_ml={}{} postfx={}{} ui={}{}] min_physics={} lane_q={} sm_cu_utilization={:.2} ai_ml_drop_rate={:.3}{}",
            self.gms_scaler.mode.as_str(),
            self.gms_scaler.target_fps,
            self.gms_scaler.guardrail.as_str(),
            self.gms_scaler.profile.as_str(),
            self.gms_scaler.auto_gpu_routing,
            self.gms_scaler.budgets.render_budget_pct,
            gpu_str(self.gms_scaler.budgets.render_gpu),
            self.gms_scaler.budgets.physics_budget_pct,
            gpu_str(self.gms_scaler.budgets.physics_gpu),
            self.gms_scaler.budgets.ai_ml_budget_pct,
            gpu_str(self.gms_scaler.budgets.ai_ml_gpu),
            self.gms_scaler.budgets.postfx_budget_pct,
            gpu_str(self.gms_scaler.budgets.postfx_gpu),
            self.gms_scaler.budgets.ui_budget_pct,
            gpu_str(self.gms_scaler.budgets.ui_gpu),
            self.gms_scaler.min_physics_budget_pct,
            self.gms_last_lane_queue_depth,
            self.gms_last_sm_cu_utilization,
            ai_ml_drop_rate,
            self.gms_last_fallback_reason
                .as_ref()
                .map(|r| format!(" fallback_reason={r}"))
                .unwrap_or_default()
        ))
    }

    pub fn mls_status_line(&self) -> String {
        self.mls.status_line()
    }

    /// Selected runtime bridge path.
    pub fn path(&self) -> RuntimeBridgePath {
        self.path
    }

    /// Submit one frame workload into active bridge path and seal frame.
    pub fn submit_scene_workload(
        &mut self,
        frame_id: u64,
        frame: &SceneFrameInstances,
        dynamic_body_count: usize,
        viewport_width: u32,
        viewport_height: u32,
    ) -> RuntimeBridgeSubmission {
        match self.path {
            RuntimeBridgePath::GmsPath => {
                let estimate = estimate_scene_workload_requests(
                    frame,
                    dynamic_body_count,
                    viewport_width,
                    viewport_height,
                    self.config.gms_scene_workload,
                );
                let estimate = self.apply_gms_scaler(estimate);
                let frame_loop = match &mut self.inner {
                    RuntimeBridgeLoop::Gms(frame_loop) => frame_loop,
                    RuntimeBridgeLoop::Mgs(_) => unreachable!("path/inner mismatch: expected GMS"),
                };
                RuntimeBridgeSubmission::Gms(submit_scene_estimate_to_bridge(
                    frame_loop,
                    frame_id,
                    &estimate,
                    self.config.gms_scene_dispatch,
                ))
            }
            RuntimeBridgePath::MgsPath => {
                let hint = estimate_mobile_workload_hint(
                    frame,
                    dynamic_body_count,
                    viewport_width,
                    viewport_height,
                    self.config.mgs_scene_workload,
                );
                let frame_loop = match &mut self.inner {
                    RuntimeBridgeLoop::Mgs(frame_loop) => frame_loop,
                    RuntimeBridgeLoop::Gms(_) => unreachable!("path/inner mismatch: expected MGS"),
                };
                RuntimeBridgeSubmission::Mgs(submit_mobile_hint_to_bridge(
                    frame_loop,
                    frame_id,
                    hint,
                    self.config.mgs_scene_dispatch,
                ))
            }
        }
    }

    /// Tick active bridge loop, pop next plan, and optionally reuse last plan as fail-soft fallback.
    pub fn tick_and_plan(&mut self) -> RuntimeBridgeTick {
        let mut tick = RuntimeBridgeTick::empty(self.path);

        let next_plan = match &mut self.inner {
            RuntimeBridgeLoop::Gms(frame_loop) => {
                let result = frame_loop.tick();
                tick.bridge_pump_published = result.bridge_published_frames;
                tick.bridge_pump_drained = result.drained_frame_plans;
                tick.queued_plan_depth = result.queued_frame_plans;
                self.gms_last_lane_queue_depth = result.queued_frame_plans;
                let ai_ml_drop_rate = 1.0
                    - (self.gms_ai_ml_kept_jobs as f32
                        / self.gms_ai_ml_requested_jobs.max(1) as f32)
                        .clamp(0.0, 1.0);
                self.mls.note_scheduler_feedback(
                    self.gms_last_lane_queue_depth,
                    self.mls.telemetry().train_queue_depth,
                    ai_ml_drop_rate,
                );
                frame_loop.pop_next_frame_plan().map(RuntimeFramePlan::Gms)
            }
            RuntimeBridgeLoop::Mgs(frame_loop) => {
                let result = frame_loop.tick();
                tick.bridge_pump_published = result.bridge_published_frames;
                tick.bridge_pump_drained = result.drained_frame_plans;
                tick.queued_plan_depth = result.queued_frame_plans;
                self.gms_last_lane_queue_depth = result.queued_frame_plans;
                self.mls.note_scheduler_feedback(
                    result.queued_frame_plans,
                    self.mls.telemetry().train_queue_depth,
                    self.mls.telemetry().drop_rate,
                );
                frame_loop.pop_next_frame_plan().map(RuntimeFramePlan::Mgs)
            }
        };

        if let Some(plan) = next_plan {
            self.last_plan = Some(plan.clone());
            tick.plan = Some(plan);
            return tick;
        }

        if self.config.allow_fallback_reuse {
            if let Some(last) = self.last_plan.clone() {
                tick.used_fallback_plan = true;
                tick.plan = Some(last);
            }
        }

        tick
    }

    /// Snapshot unified path-independent metrics.
    pub fn metrics(&self) -> RuntimeBridgeMetrics {
        let mls = self.mls.telemetry();
        match &self.inner {
            RuntimeBridgeLoop::Gms(frame_loop) => {
                let m = frame_loop.metrics();
                RuntimeBridgeMetrics {
                    bridge_path: self.path,
                    queued_plan_depth: m.queued_frame_plans,
                    bridge_tick_published_frames: m.bridge_tick_published_frames,
                    bridge_tick_drained_plans: m.bridge_tick_drained_plans,
                    frame_plans_popped: m.frame_plans_popped,
                    gms_mode: Some(self.gms_scaler.mode),
                    domain_budgets: Some(self.gms_scaler.budgets),
                    performance_profile: self.gms_scaler.profile,
                    sm_cu_utilization: self.gms_last_sm_cu_utilization,
                    lane_queue_depth: self.gms_last_lane_queue_depth,
                    ai_ml_drop_rate: 1.0
                        - (self.gms_ai_ml_kept_jobs as f32
                            / self.gms_ai_ml_requested_jobs.max(1) as f32)
                            .clamp(0.0, 1.0),
                    fallback_reason: self.gms_last_fallback_reason.clone(),
                    mls_backend: mls.backend,
                    mls_device: mls.device.clone(),
                    mls_active_workloads: mls.active_workloads.clone(),
                    mls_infer_queue_depth: mls.infer_queue_depth,
                    mls_train_queue_depth: mls.train_queue_depth,
                    mls_drop_rate: mls.drop_rate,
                    mls_fallback_reason: mls
                        .fallback_reason
                        .map(|reason| reason.as_str().to_string()),
                    mls_precision: mls.precision,
                    mls_step_time_ms: mls.step_time_ms,
                }
            }
            RuntimeBridgeLoop::Mgs(frame_loop) => {
                let m = frame_loop.metrics();
                RuntimeBridgeMetrics {
                    bridge_path: self.path,
                    queued_plan_depth: m.queued_frame_plans,
                    bridge_tick_published_frames: m.bridge_tick_published_frames,
                    bridge_tick_drained_plans: m.bridge_tick_drained_plans,
                    frame_plans_popped: m.frame_plans_popped,
                    gms_mode: None,
                    domain_budgets: None,
                    performance_profile: self.gms_scaler.profile,
                    sm_cu_utilization: 0.0,
                    lane_queue_depth: m.queued_frame_plans,
                    ai_ml_drop_rate: 0.0,
                    fallback_reason: None,
                    mls_backend: mls.backend,
                    mls_device: mls.device.clone(),
                    mls_active_workloads: mls.active_workloads.clone(),
                    mls_infer_queue_depth: mls.infer_queue_depth,
                    mls_train_queue_depth: mls.train_queue_depth,
                    mls_drop_rate: mls.drop_rate,
                    mls_fallback_reason: mls
                        .fallback_reason
                        .map(|reason| reason.as_str().to_string()),
                    mls_precision: mls.precision,
                    mls_step_time_ms: mls.step_time_ms,
                }
            }
        }
    }

    fn apply_gms_scaler(&mut self, mut estimate: SceneWorkloadEstimate) -> SceneWorkloadEstimate {
        let budgets = self.gms_scaler.budgets;
        let mut fallback_reason: Option<String> = None;

        self.gms_complexity_ema = if self.gms_complexity_ema <= f64::EPSILON {
            estimate.complexity_score
        } else {
            self.gms_complexity_ema + (estimate.complexity_score - self.gms_complexity_ema) * 0.18
        };

        let requested_sampled = estimate.multi_gpu.sampled_processing_jobs;
        let requested_object = estimate.multi_gpu.object_updates;
        let requested_physics = estimate.multi_gpu.physics_jobs;
        let requested_ai_ml = estimate.multi_gpu.ai_ml_jobs;
        let requested_postfx = estimate.multi_gpu.post_fx_jobs;
        let requested_ui = estimate.multi_gpu.ui_jobs;
        let total_jobs = requested_sampled
            .saturating_add(requested_object)
            .saturating_add(requested_physics)
            .saturating_add(requested_ai_ml)
            .saturating_add(requested_postfx)
            .saturating_add(requested_ui);
        if total_jobs == 0 {
            self.gms_last_sm_cu_utilization = 1.0;
            self.gms_last_fallback_reason = None;
            return estimate;
        }

        let target_frame_ms = 1_000.0 / self.gms_scaler.target_fps.max(1) as f64;
        let source_frame_ms = estimate.multi_gpu.target_frame_budget_ms.max(0.1);
        let queue_pressure = (self.gms_last_lane_queue_depth as f64 / 3.0).clamp(0.0, 2.0);
        let complexity_pressure = (self.gms_complexity_ema / 5_500.0).clamp(0.20, 3.0);
        let mut pressure = (source_frame_ms / target_frame_ms).clamp(0.25, 3.0);
        pressure = pressure + queue_pressure * 0.22 + (complexity_pressure - 1.0) * 0.24;
        pressure = pressure.clamp(0.25, 3.0);

        let mut render_budget = budgets.render_budget_pct as f64;
        let mut physics_budget = budgets.physics_budget_pct as f64;
        let mut ai_ml_budget = budgets.ai_ml_budget_pct as f64;
        let mut postfx_budget = budgets.postfx_budget_pct as f64;
        let mut ui_budget = budgets.ui_budget_pct as f64;
        let sum = render_budget + physics_budget + ai_ml_budget + postfx_budget + ui_budget;
        if sum > 100.0 {
            let inv = 100.0 / sum;
            render_budget *= inv;
            physics_budget *= inv;
            ai_ml_budget *= inv;
            postfx_budget *= inv;
            ui_budget *= inv;
        } else if sum < 100.0 {
            render_budget += 100.0 - sum;
        }

        let render_cap = ((total_jobs as f64) * (render_budget / 100.0)).round() as u32;
        let mut physics_cap = ((total_jobs as f64) * (physics_budget / 100.0)).round() as u32;
        let mut ai_ml_cap = ((total_jobs as f64) * (ai_ml_budget / 100.0)).round() as u32;
        let mut postfx_cap = ((total_jobs as f64) * (postfx_budget / 100.0)).round() as u32;
        let mut ui_cap = ((total_jobs as f64) * (ui_budget / 100.0)).round() as u32;
        let min_physics_cap = ((total_jobs as f64)
            * (self.gms_scaler.min_physics_budget_pct.min(100) as f64 / 100.0))
            .round() as u32;
        physics_cap = physics_cap.max(min_physics_cap);

        if matches!(self.gms_scaler.mode, GmsScalerMode::Adaptive) && pressure > 1.05 {
            let (postfx_scale, ai_ml_scale) = match self.gms_scaler.guardrail {
                GmsGuardrailProfile::Aggressive => (0.45, 0.58),
                GmsGuardrailProfile::Balanced => (0.62, 0.74),
                GmsGuardrailProfile::Relaxed => (0.80, 0.88),
            };
            let pressure_scale = (1.0 / pressure).clamp(0.35, 1.0);
            postfx_cap = ((postfx_cap as f64) * pressure_scale * postfx_scale).round() as u32;
            let postfx_trimmed_first = requested_postfx > postfx_cap.max(1);
            if !postfx_trimmed_first || pressure > 1.35 {
                ai_ml_cap = ((ai_ml_cap as f64) * pressure_scale * ai_ml_scale).round() as u32;
            }
            if pressure > 1.25 {
                let ui_scale = (pressure_scale * 0.92).clamp(0.45, 1.0);
                ui_cap = ((ui_cap as f64) * ui_scale).round() as u32;
            }
            let clipped_lanes = if postfx_trimmed_first && pressure <= 1.35 {
                "postfx"
            } else {
                "postfx+ai_ml"
            };
            fallback_reason = Some(format!(
                "adaptive guardrail clipped {clipped_lanes} (pressure={pressure:.2})"
            ));
        }

        let mut new_sampled = requested_sampled;
        let mut new_object = requested_object;
        let render_current = requested_sampled.saturating_add(requested_object);
        if matches!(self.gms_scaler.mode, GmsScalerMode::Fixed)
            || (matches!(self.gms_scaler.mode, GmsScalerMode::Adaptive)
                && render_current > render_cap)
        {
            if render_current > render_cap && render_current > 0 {
                let ratio = render_cap as f64 / render_current as f64;
                new_sampled = ((requested_sampled as f64) * ratio).round() as u32;
                new_object = ((requested_object as f64) * ratio).round() as u32;
            }
        }

        let new_physics = if matches!(self.gms_scaler.mode, GmsScalerMode::Fixed) {
            requested_physics.min(physics_cap.max(1))
        } else {
            requested_physics
        };
        let new_ai_ml = requested_ai_ml.min(ai_ml_cap.max(1));
        let new_postfx = requested_postfx.min(postfx_cap.max(1));
        let new_ui = requested_ui.min(ui_cap.max(1));

        self.gms_ai_ml_requested_jobs = self
            .gms_ai_ml_requested_jobs
            .saturating_add(requested_ai_ml as u64);
        self.gms_ai_ml_kept_jobs = self.gms_ai_ml_kept_jobs.saturating_add(new_ai_ml as u64);
        let ai_ml_drop_rate = 1.0
            - (self.gms_ai_ml_kept_jobs as f32 / self.gms_ai_ml_requested_jobs.max(1) as f32)
                .clamp(0.0, 1.0);
        self.mls
            .note_scheduler_feedback(self.gms_last_lane_queue_depth, 0, ai_ml_drop_rate);

        let kept_total = new_sampled
            .saturating_add(new_object)
            .saturating_add(new_physics)
            .saturating_add(new_ai_ml)
            .saturating_add(new_postfx)
            .saturating_add(new_ui);
        self.gms_last_sm_cu_utilization = (kept_total as f32 / total_jobs as f32).clamp(0.0, 1.0);

        if kept_total < total_jobs && fallback_reason.is_none() {
            fallback_reason = Some(format!(
                "scaled domains from {} to {} jobs",
                total_jobs, kept_total
            ));
        }
        self.gms_last_fallback_reason = fallback_reason;

        estimate.multi_gpu.sampled_processing_jobs = new_sampled;
        estimate.multi_gpu.object_updates = new_object;
        estimate.multi_gpu.physics_jobs = new_physics;
        estimate.multi_gpu.ai_ml_jobs = new_ai_ml;
        estimate.multi_gpu.post_fx_jobs = new_postfx;
        estimate.multi_gpu.ui_jobs = new_ui;

        estimate.single_gpu.object_updates = new_object
            .saturating_add(new_ui / 2)
            .saturating_add(new_ai_ml / 3);
        estimate.single_gpu.physics_jobs = new_physics
            .saturating_add(new_postfx / 3)
            .saturating_add(new_ai_ml / 2);
        estimate
    }
}

/// Map scheduler policy to runtime bridge path.
pub fn runtime_bridge_path_from_scheduler(scheduler: GraphicsSchedulerPath) -> RuntimeBridgePath {
    match scheduler {
        GraphicsSchedulerPath::Gms => RuntimeBridgePath::GmsPath,
        GraphicsSchedulerPath::Mgs => RuntimeBridgePath::MgsPath,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gms::{MultiGpuWorkloadRequest, WorkloadRequest};
    use std::time::Duration;

    #[test]
    fn scheduler_mapping_prefers_matching_path() {
        assert_eq!(
            runtime_bridge_path_from_scheduler(GraphicsSchedulerPath::Gms),
            RuntimeBridgePath::GmsPath
        );
        assert_eq!(
            runtime_bridge_path_from_scheduler(GraphicsSchedulerPath::Mgs),
            RuntimeBridgePath::MgsPath
        );
    }

    #[test]
    fn tick_reuses_last_plan_when_fallback_enabled() {
        let mut orchestrator = RuntimeBridgeOrchestrator::new_for_scheduler(
            GraphicsSchedulerPath::Mgs,
            "Mali-G610",
            RuntimeBridgeConfig::default(),
            1280,
            720,
        );

        let frame = SceneFrameInstances::default();
        let _submission = orchestrator.submit_scene_workload(1, &frame, 128, 1280, 720);
        let mut first_tick = orchestrator.tick_and_plan();
        for _ in 0..128 {
            if first_tick.plan.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
            first_tick = orchestrator.tick_and_plan();
        }
        assert!(
            first_tick.plan.is_some(),
            "expected at least one plan after submission/seal"
        );

        // No submission for frame 2, so fallback should reuse latest valid plan.
        let second_tick = orchestrator.tick_and_plan();
        assert!(second_tick.plan.is_some());
        assert!(second_tick.used_fallback_plan);
    }

    #[test]
    fn adaptive_gms_guardrail_trims_postfx_before_ai_ml() {
        let mut orchestrator = RuntimeBridgeOrchestrator::new_for_scheduler(
            GraphicsSchedulerPath::Mgs,
            "Mali-G610",
            RuntimeBridgeConfig::default(),
            1280,
            720,
        );
        orchestrator.set_gms_guardrail(GmsGuardrailProfile::Balanced);

        let estimate = SceneWorkloadEstimate {
            single_gpu: WorkloadRequest::default(),
            multi_gpu: MultiGpuWorkloadRequest {
                sampled_processing_jobs: 100,
                object_updates: 100,
                physics_jobs: 200,
                ai_ml_jobs: 120,
                post_fx_jobs: 160,
                ui_jobs: 20,
                target_frame_budget_ms: 20.0,
                ..MultiGpuWorkloadRequest::default()
            },
            complexity_score: 5_500.0,
            estimated_frame_bytes: 8 * 1024 * 1024,
        };

        let scaled = orchestrator.apply_gms_scaler(estimate);

        assert!(
            scaled.multi_gpu.post_fx_jobs < estimate.multi_gpu.post_fx_jobs,
            "postfx should absorb moderate adaptive pressure first"
        );
        assert_eq!(
            scaled.multi_gpu.ai_ml_jobs, estimate.multi_gpu.ai_ml_jobs,
            "ai/ml should stay intact while postfx can absorb the spike"
        );
        assert!(orchestrator
            .gms_last_fallback_reason
            .as_deref()
            .unwrap_or_default()
            .contains("clipped postfx"));
    }

    // E3 calibration gate tests

    fn make_gms_orchestrator() -> RuntimeBridgeOrchestrator {
        RuntimeBridgeOrchestrator::new_for_scheduler(
            GraphicsSchedulerPath::Gms,
            "RTX 4090",
            RuntimeBridgeConfig::default(),
            1920,
            1080,
        )
    }

    fn make_estimate(physics: u32, ai_ml: u32, postfx: u32, frame_ms: f64) -> SceneWorkloadEstimate {
        SceneWorkloadEstimate {
            single_gpu: WorkloadRequest::default(),
            multi_gpu: MultiGpuWorkloadRequest {
                sampled_processing_jobs: 100,
                object_updates: 100,
                physics_jobs: physics,
                ai_ml_jobs: ai_ml,
                post_fx_jobs: postfx,
                ui_jobs: 20,
                target_frame_budget_ms: frame_ms,
                ..MultiGpuWorkloadRequest::default()
            },
            complexity_score: 5_500.0,
            estimated_frame_bytes: 8 * 1024 * 1024,
        }
    }

    #[test]
    fn gms_adaptive_mode_never_reduces_physics_jobs() {
        let mut orchestrator = make_gms_orchestrator();
        orchestrator.set_gms_mode(GmsScalerMode::Adaptive);
        // Simulate a heavily loaded queue to maximize pressure.
        orchestrator.gms_last_lane_queue_depth = 10;

        let estimate = make_estimate(800, 300, 300, 40.0); // 40ms at 60fps target = 2.4× pressure
        let scaled = orchestrator.apply_gms_scaler(estimate);

        assert_eq!(
            scaled.multi_gpu.physics_jobs, estimate.multi_gpu.physics_jobs,
            "adaptive mode must never reduce physics jobs regardless of pressure"
        );
    }

    #[test]
    fn gms_fixed_mode_physics_cap_respects_min_physics_budget() {
        let mut orchestrator = make_gms_orchestrator();
        orchestrator.set_gms_mode(GmsScalerMode::Fixed);
        // Set a very low physics budget percentage so min_physics_budget_pct kicks in.
        orchestrator.set_gms_budget(GmsScalerDomain::Physics, 5);

        let total_jobs: u32 = 100 + 100 + 800 + 300 + 300 + 20;
        let min_physics_pct = orchestrator.gms_scaler.min_physics_budget_pct as f64 / 100.0;
        let min_physics_cap = (total_jobs as f64 * min_physics_pct).round() as u32;

        let estimate = make_estimate(800, 300, 300, 16.67);
        let scaled = orchestrator.apply_gms_scaler(estimate);

        assert!(
            scaled.multi_gpu.physics_jobs >= min_physics_cap.min(estimate.multi_gpu.physics_jobs),
            "fixed mode physics output must respect min_physics_budget_pct floor (expected >= {min_physics_cap}, got {})",
            scaled.multi_gpu.physics_jobs
        );
    }

    #[test]
    fn gms_scaler_is_deterministic_under_fixed_inputs() {
        let estimate = make_estimate(400, 200, 250, 25.0);

        let mut a = make_gms_orchestrator();
        a.set_gms_mode(GmsScalerMode::Adaptive);
        let out_a = a.apply_gms_scaler(estimate.clone());

        let mut b = make_gms_orchestrator();
        b.set_gms_mode(GmsScalerMode::Adaptive);
        let out_b = b.apply_gms_scaler(estimate.clone());

        assert_eq!(
            out_a.multi_gpu.physics_jobs, out_b.multi_gpu.physics_jobs,
            "physics output must be deterministic"
        );
        assert_eq!(
            out_a.multi_gpu.ai_ml_jobs, out_b.multi_gpu.ai_ml_jobs,
            "ai_ml output must be deterministic"
        );
        assert_eq!(
            out_a.multi_gpu.post_fx_jobs, out_b.multi_gpu.post_fx_jobs,
            "postfx output must be deterministic"
        );
    }

    #[test]
    fn gms_guardrail_profiles_cut_in_correct_order() {
        // Same high pressure — Aggressive must cut postfx more than Balanced, Balanced more than Relaxed.
        let estimate = make_estimate(300, 200, 300, 35.0); // ~2× pressure

        let mut aggressive = make_gms_orchestrator();
        aggressive.set_gms_mode(GmsScalerMode::Adaptive);
        aggressive.set_gms_guardrail(GmsGuardrailProfile::Aggressive);
        let out_aggressive = aggressive.apply_gms_scaler(estimate.clone());

        let mut balanced = make_gms_orchestrator();
        balanced.set_gms_mode(GmsScalerMode::Adaptive);
        balanced.set_gms_guardrail(GmsGuardrailProfile::Balanced);
        let out_balanced = balanced.apply_gms_scaler(estimate.clone());

        let mut relaxed = make_gms_orchestrator();
        relaxed.set_gms_mode(GmsScalerMode::Adaptive);
        relaxed.set_gms_guardrail(GmsGuardrailProfile::Relaxed);
        let out_relaxed = relaxed.apply_gms_scaler(estimate.clone());

        assert!(
            out_aggressive.multi_gpu.post_fx_jobs <= out_balanced.multi_gpu.post_fx_jobs,
            "aggressive must cut postfx at least as much as balanced (aggressive={}, balanced={})",
            out_aggressive.multi_gpu.post_fx_jobs,
            out_balanced.multi_gpu.post_fx_jobs
        );
        assert!(
            out_balanced.multi_gpu.post_fx_jobs <= out_relaxed.multi_gpu.post_fx_jobs,
            "balanced must cut postfx at least as much as relaxed (balanced={}, relaxed={})",
            out_balanced.multi_gpu.post_fx_jobs,
            out_relaxed.multi_gpu.post_fx_jobs
        );
    }

    #[test]
    fn gms_complexity_ema_converges_toward_target_score() {
        let mut orchestrator = make_gms_orchestrator();
        orchestrator.set_gms_mode(GmsScalerMode::Adaptive);

        let target_complexity = 8_000.0_f64;
        let estimate = SceneWorkloadEstimate {
            single_gpu: WorkloadRequest::default(),
            multi_gpu: MultiGpuWorkloadRequest {
                sampled_processing_jobs: 100,
                object_updates: 100,
                physics_jobs: 200,
                ai_ml_jobs: 100,
                post_fx_jobs: 100,
                ui_jobs: 10,
                target_frame_budget_ms: 16.67,
                ..MultiGpuWorkloadRequest::default()
            },
            complexity_score: target_complexity,
            estimated_frame_bytes: 4 * 1024 * 1024,
        };

        for _ in 0..30 {
            orchestrator.apply_gms_scaler(estimate.clone());
        }

        // After 30 frames at 0.18 EMA coefficient, EMA should exceed 98% of target.
        // 1 - (1 - 0.18)^30 ≈ 0.995
        let expected_min = target_complexity * 0.98;
        assert!(
            orchestrator.gms_complexity_ema >= expected_min,
            "EMA must converge toward target after 30 frames (got {:.1}, expected >= {expected_min:.1})",
            orchestrator.gms_complexity_ema
        );
    }

    #[test]
    fn gms_ai_ml_drop_rate_reflects_clipping_ratio() {
        let mut orchestrator = make_gms_orchestrator();
        orchestrator.set_gms_mode(GmsScalerMode::Adaptive);
        orchestrator.set_gms_guardrail(GmsGuardrailProfile::Aggressive);
        orchestrator.gms_last_lane_queue_depth = 8; // force high pressure

        let estimate = make_estimate(300, 500, 300, 40.0); // heavy ai_ml under pressure
        for _ in 0..10 {
            orchestrator.apply_gms_scaler(estimate.clone());
        }

        let requested = orchestrator.gms_ai_ml_requested_jobs.max(1);
        let kept = orchestrator.gms_ai_ml_kept_jobs;
        let drop_rate = 1.0 - (kept as f64 / requested as f64).clamp(0.0, 1.0);

        let metric = orchestrator
            .gms_metric("ai_ml_drop_rate")
            .expect("metric must be present");

        assert!(
            (metric - drop_rate).abs() < 1e-9,
            "gms_metric('ai_ml_drop_rate') must match internal ratio"
        );
        // Under aggressive profile and high pressure, we expect some ai/ml dropping.
        assert!(
            drop_rate > 0.0,
            "expected ai/ml drop > 0 under aggressive profile + high pressure"
        );
    }
}
