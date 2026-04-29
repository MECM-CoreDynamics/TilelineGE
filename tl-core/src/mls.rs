//! Machine Learning Scaler (MLS) runtime contract.
//!
//! `MLS` is the engine-owned ML execution surface shared by runtime, GMS, and
//! future native backend adapters. The v0.6.0 foundation is intentionally
//! fail-soft: unsupported native paths degrade to CPU fallback or workload
//! disablement instead of crashing or stalling the frame loop.

use std::collections::BTreeMap;

/// Runtime execution mode for MLS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlsExecutionMode {
    Off,
    #[default]
    Auto,
    On,
}

impl MlsExecutionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "auto" => Some(Self::Auto),
            "on" => Some(Self::On),
            _ => None,
        }
    }
}

/// Selected or requested MLS backend family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum MlsBackendKind {
    Auto,
    Amd,
    Nvidia,
    Apple,
    Rockchip,
    #[default]
    Cpu,
}

impl MlsBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Amd => "amd",
            Self::Nvidia => "nvidia",
            Self::Apple => "apple",
            Self::Rockchip => "rockchip",
            Self::Cpu => "cpu",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "amd" | "rocm" | "hip" => Some(Self::Amd),
            "nvidia" | "cuda" => Some(Self::Nvidia),
            "apple" | "ane" | "coreml" | "core_ml" => Some(Self::Apple),
            "rockchip" | "rknn" | "npu" => Some(Self::Rockchip),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Preferred numeric precision for MLS workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum MlsPrecisionMode {
    #[default]
    Auto,
    Fp32,
    Fp16,
    Bf16,
    Int8,
}

impl MlsPrecisionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "fp32" => Some(Self::Fp32),
            "fp16" => Some(Self::Fp16),
            "bf16" => Some(Self::Bf16),
            "int8" => Some(Self::Int8),
            _ => None,
        }
    }
}

/// Canonical MLS workload lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MlsWorkloadKind {
    Upscale,
    Agent,
    PhysicsAssist,
    Training,
}

impl MlsWorkloadKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upscale => "upscale",
            Self::Agent => "agent",
            Self::PhysicsAssist => "physics_assist",
            Self::Training => "training",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "upscale" => Some(Self::Upscale),
            "agent" | "agents" => Some(Self::Agent),
            "physics_assist" | "physics-assist" | "physicsassist" => Some(Self::PhysicsAssist),
            "training" | "train" => Some(Self::Training),
            _ => None,
        }
    }
}

/// Why MLS fell back or disabled a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlsFallbackReason {
    DisabledByMode,
    UnsupportedBackend,
    NativeAdapterUnavailable,
    CpuFallback,
    MissingModelArtifact,
    TrainingUnsupported,
    UnsupportedWorkload,
    TrainingDisabled,
}

impl MlsFallbackReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DisabledByMode => "disabled_by_mode",
            Self::UnsupportedBackend => "unsupported_backend",
            Self::NativeAdapterUnavailable => "native_adapter_unavailable",
            Self::CpuFallback => "cpu_fallback",
            Self::MissingModelArtifact => "missing_model_artifact",
            Self::TrainingUnsupported => "training_unsupported",
            Self::UnsupportedWorkload => "unsupported_workload",
            Self::TrainingDisabled => "training_disabled",
        }
    }
}

/// Static capability matrix reported by one MLS backend/device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsCapabilityMatrix {
    pub supports_inference: bool,
    pub supports_training: bool,
    pub supports_upscale: bool,
    pub supports_agent: bool,
    pub supports_physics_assist: bool,
    pub supports_checkpointing: bool,
    pub supported_precisions: Vec<MlsPrecisionMode>,
    pub uses_pak_model_container: bool,
}

impl Default for MlsCapabilityMatrix {
    fn default() -> Self {
        Self {
            supports_inference: true,
            supports_training: false,
            supports_upscale: true,
            supports_agent: true,
            supports_physics_assist: true,
            supports_checkpointing: false,
            supported_precisions: vec![MlsPrecisionMode::Fp32],
            uses_pak_model_container: true,
        }
    }
}

/// Device profile resolved for the active MLS adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsDeviceProfile {
    pub backend: MlsBackendKind,
    pub device_name: String,
    pub device_class: String,
    pub matrix_engine: String,
    pub capability_matrix: MlsCapabilityMatrix,
}

/// Shipping model artifact reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsModelArtifact {
    pub slot: u32,
    pub source: String,
    pub from_pak: bool,
}

/// One inference request issued by runtime or script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsInferenceRequest {
    pub slot: u32,
    pub workload: MlsWorkloadKind,
    pub batch_hint: u32,
}

/// One training request issued by runtime or script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsTrainingRequest {
    pub slot: u32,
    pub steps: u32,
}

/// Local training configuration for v0.6.0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsTrainingConfig {
    pub allow_training: bool,
    pub local_single_device_only: bool,
    pub max_steps_per_batch: u32,
}

impl Default for MlsTrainingConfig {
    fn default() -> Self {
        Self {
            allow_training: false,
            local_single_device_only: true,
            max_steps_per_batch: 512,
        }
    }
}

/// Workload enablement and budget splits inside the top-level `ai_ml` lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlsWorkloadBudgetProfile {
    pub upscale_budget_pct: u8,
    pub agent_budget_pct: u8,
    pub physics_assist_budget_pct: u8,
    pub training_budget_pct: u8,
}

impl Default for MlsWorkloadBudgetProfile {
    fn default() -> Self {
        Self {
            upscale_budget_pct: 40,
            agent_budget_pct: 25,
            physics_assist_budget_pct: 20,
            training_budget_pct: 15,
        }
    }
}

/// Public MLS runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlsRuntimeConfig {
    pub mode: MlsExecutionMode,
    pub backend: MlsBackendKind,
    pub precision: MlsPrecisionMode,
    pub model_pack: Option<String>,
    pub allow_training: bool,
    pub upscale_enabled: bool,
    pub agent_enabled: bool,
    pub physics_assist_enabled: bool,
    pub training_enabled: bool,
    pub budgets: MlsWorkloadBudgetProfile,
}

impl Default for MlsRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: MlsExecutionMode::Auto,
            backend: MlsBackendKind::Auto,
            precision: MlsPrecisionMode::Auto,
            model_pack: None,
            allow_training: false,
            upscale_enabled: true,
            agent_enabled: true,
            physics_assist_enabled: true,
            training_enabled: false,
            budgets: MlsWorkloadBudgetProfile::default(),
        }
    }
}

impl MlsRuntimeConfig {
    pub fn set_budget(&mut self, workload: MlsWorkloadKind, pct: u8) {
        let pct = pct.min(100);
        match workload {
            MlsWorkloadKind::Upscale => self.budgets.upscale_budget_pct = pct,
            MlsWorkloadKind::Agent => self.budgets.agent_budget_pct = pct,
            MlsWorkloadKind::PhysicsAssist => self.budgets.physics_assist_budget_pct = pct,
            MlsWorkloadKind::Training => self.budgets.training_budget_pct = pct,
        }
    }

    pub fn workload_enabled(&self, workload: MlsWorkloadKind) -> bool {
        match workload {
            MlsWorkloadKind::Upscale => self.upscale_enabled,
            MlsWorkloadKind::Agent => self.agent_enabled,
            MlsWorkloadKind::PhysicsAssist => self.physics_assist_enabled,
            MlsWorkloadKind::Training => self.training_enabled,
        }
    }
}

/// Live MLS telemetry exported to runtime/console/script.
#[derive(Debug, Clone, PartialEq)]
pub struct MlsTelemetry {
    pub backend: MlsBackendKind,
    pub device: String,
    pub active_workloads: Vec<MlsWorkloadKind>,
    pub infer_queue_depth: usize,
    pub train_queue_depth: usize,
    pub drop_rate: f32,
    pub fallback_reason: Option<MlsFallbackReason>,
    pub precision: MlsPrecisionMode,
    pub step_time_ms: f32,
}

impl Default for MlsTelemetry {
    fn default() -> Self {
        Self {
            backend: MlsBackendKind::Cpu,
            device: "cpu".to_string(),
            active_workloads: Vec::new(),
            infer_queue_depth: 0,
            train_queue_depth: 0,
            drop_rate: 0.0,
            fallback_reason: None,
            precision: MlsPrecisionMode::Fp32,
            step_time_ms: 0.0,
        }
    }
}

/// Backend adapter contract.
pub trait MlsBackendAdapter: Send + Sync {
    fn kind(&self) -> MlsBackendKind;
    fn is_available(&self) -> bool;
    fn profile(&self) -> MlsDeviceProfile;
    fn supports_workload(&self, workload: MlsWorkloadKind) -> bool;
}

#[derive(Debug, Clone)]
struct StaticMlsAdapter {
    kind: MlsBackendKind,
    available: bool,
    profile: MlsDeviceProfile,
}

impl StaticMlsAdapter {
    fn new(
        kind: MlsBackendKind,
        available: bool,
        device_name: &str,
        device_class: &str,
        matrix_engine: &str,
        capability_matrix: MlsCapabilityMatrix,
    ) -> Self {
        Self {
            kind,
            available,
            profile: MlsDeviceProfile {
                backend: kind,
                device_name: device_name.to_string(),
                device_class: device_class.to_string(),
                matrix_engine: matrix_engine.to_string(),
                capability_matrix,
            },
        }
    }
}

impl MlsBackendAdapter for StaticMlsAdapter {
    fn kind(&self) -> MlsBackendKind {
        self.kind
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn profile(&self) -> MlsDeviceProfile {
        self.profile.clone()
    }

    fn supports_workload(&self, workload: MlsWorkloadKind) -> bool {
        let caps = &self.profile.capability_matrix;
        match workload {
            MlsWorkloadKind::Upscale => caps.supports_upscale && caps.supports_inference,
            MlsWorkloadKind::Agent => caps.supports_agent && caps.supports_inference,
            MlsWorkloadKind::PhysicsAssist => {
                caps.supports_physics_assist && caps.supports_inference
            }
            MlsWorkloadKind::Training => caps.supports_training,
        }
    }
}

/// Fail-soft runtime shell around the MLS contract.
pub struct MlsRuntime {
    config: MlsRuntimeConfig,
    training: MlsTrainingConfig,
    telemetry: MlsTelemetry,
    adapters: Vec<Box<dyn MlsBackendAdapter>>,
    bound_models: BTreeMap<u32, MlsModelArtifact>,
    checkpoints: BTreeMap<String, u32>,
}

impl MlsRuntime {
    pub fn new(config: MlsRuntimeConfig) -> Self {
        Self::with_adapters(config, default_adapters())
    }

    pub fn with_adapters(
        config: MlsRuntimeConfig,
        adapters: Vec<Box<dyn MlsBackendAdapter>>,
    ) -> Self {
        let training = MlsTrainingConfig {
            allow_training: config.allow_training,
            ..MlsTrainingConfig::default()
        };
        let mut runtime = Self {
            config,
            training,
            telemetry: MlsTelemetry::default(),
            adapters,
            bound_models: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
        };
        runtime.refresh_backend();
        runtime
    }

    pub fn config(&self) -> &MlsRuntimeConfig {
        &self.config
    }

    pub fn telemetry(&self) -> &MlsTelemetry {
        &self.telemetry
    }

    pub fn set_mode(&mut self, mode: MlsExecutionMode) {
        self.config.mode = mode;
        self.refresh_backend();
    }

    pub fn set_backend(&mut self, backend: MlsBackendKind) {
        self.config.backend = backend;
        self.refresh_backend();
    }

    pub fn set_precision(&mut self, precision: MlsPrecisionMode) {
        self.config.precision = precision;
        self.refresh_backend();
    }

    pub fn set_model_pack(&mut self, path: Option<String>) {
        self.config.model_pack = path;
    }

    pub fn set_allow_training(&mut self, allow: bool) {
        self.config.allow_training = allow;
        self.training.allow_training = allow;
        self.refresh_backend();
    }

    pub fn set_workload_enabled(&mut self, workload: MlsWorkloadKind, enabled: bool) {
        match workload {
            MlsWorkloadKind::Upscale => self.config.upscale_enabled = enabled,
            MlsWorkloadKind::Agent => self.config.agent_enabled = enabled,
            MlsWorkloadKind::PhysicsAssist => self.config.physics_assist_enabled = enabled,
            MlsWorkloadKind::Training => self.config.training_enabled = enabled,
        }
        self.refresh_backend();
    }

    pub fn set_budget(&mut self, workload: MlsWorkloadKind, pct: u8) {
        self.config.set_budget(workload, pct);
    }

    pub fn bind_model(&mut self, slot: u32, source: impl Into<String>) -> Result<String, String> {
        let source = source.into();
        let from_pak = source.ends_with(".pak")
            || source.contains(".pak#")
            || self
                .config
                .model_pack
                .as_ref()
                .map(|pack| source.starts_with(pack))
                .unwrap_or(false);
        self.bound_models.insert(
            slot,
            MlsModelArtifact {
                slot,
                source: source.clone(),
                from_pak,
            },
        );
        Ok(format!(
            "mls model slot {slot} bound to '{}'{}",
            source,
            if from_pak { " (pak)" } else { "" }
        ))
    }

    pub fn run(&mut self, slot: u32) -> Result<String, String> {
        self.ensure_model(slot)?;
        self.ensure_runtime_ready(MlsWorkloadKind::Agent)?;
        self.telemetry.step_time_ms = 0.35;
        Ok(format!(
            "mls run slot {slot} on {} ({})",
            self.telemetry.device,
            self.telemetry.backend.as_str()
        ))
    }

    pub fn train_step(&mut self, slot: u32, steps: u32) -> Result<String, String> {
        self.ensure_model(slot)?;
        self.ensure_runtime_ready(MlsWorkloadKind::Training)?;
        let steps = steps.max(1).min(self.training.max_steps_per_batch);
        self.telemetry.step_time_ms = (steps as f32 * 0.08).clamp(0.08, 8.0);
        self.telemetry.train_queue_depth = 0;
        Ok(format!(
            "mls train slot {slot} steps={steps} backend={} device={}",
            self.telemetry.backend.as_str(),
            self.telemetry.device
        ))
    }

    pub fn checkpoint(&mut self, slot: u32, save: bool, name: &str) -> Result<String, String> {
        self.ensure_model(slot)?;
        let Some(adapter) = self.active_adapter() else {
            return Err("mls checkpoint unavailable: no active adapter".to_string());
        };
        if !adapter.profile().capability_matrix.supports_checkpointing {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::UnsupportedWorkload);
            return Err("mls checkpoint unavailable on active backend".to_string());
        }
        let key = format!("{slot}:{name}");
        if save {
            self.checkpoints.insert(key, slot);
            Ok(format!("mls checkpoint saved: slot={slot} name={name}"))
        } else if self.checkpoints.contains_key(&key) {
            Ok(format!("mls checkpoint loaded: slot={slot} name={name}"))
        } else {
            Err(format!("mls checkpoint '{name}' not found for slot {slot}"))
        }
    }

    pub fn metric(&self, name: &str) -> Option<f64> {
        let key = name.trim().to_ascii_lowercase();
        match key.as_str() {
            "infer_queue_depth" | "queue_depth" => Some(self.telemetry.infer_queue_depth as f64),
            "train_queue_depth" => Some(self.telemetry.train_queue_depth as f64),
            "drop_rate" => Some(self.telemetry.drop_rate as f64),
            "step_time_ms" => Some(self.telemetry.step_time_ms as f64),
            "backend" => Some(self.telemetry.backend as u8 as f64),
            "precision" => Some(self.telemetry.precision as u8 as f64),
            "active_workloads" => Some(self.telemetry.active_workloads.len() as f64),
            "upscale_budget_pct" => Some(self.config.budgets.upscale_budget_pct as f64),
            "agent_budget_pct" => Some(self.config.budgets.agent_budget_pct as f64),
            "physics_assist_budget_pct" => {
                Some(self.config.budgets.physics_assist_budget_pct as f64)
            }
            "training_budget_pct" => Some(self.config.budgets.training_budget_pct as f64),
            _ => None,
        }
    }

    pub fn status_line(&self) -> String {
        let active = if self.telemetry.active_workloads.is_empty() {
            "none".to_string()
        } else {
            self.telemetry
                .active_workloads
                .iter()
                .map(|workload| workload.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let fallback = self
            .telemetry
            .fallback_reason
            .map(|reason| reason.as_str().to_string())
            .unwrap_or_else(|| "none".to_string());
        format!(
            "mls | mode={} backend={} device={} precision={} workloads={} infer_q={} train_q={} drop_rate={:.3} step_ms={:.2} fallback={}",
            self.config.mode.as_str(),
            self.telemetry.backend.as_str(),
            self.telemetry.device,
            self.telemetry.precision.as_str(),
            active,
            self.telemetry.infer_queue_depth,
            self.telemetry.train_queue_depth,
            self.telemetry.drop_rate,
            self.telemetry.step_time_ms,
            fallback
        )
    }

    pub fn note_scheduler_feedback(
        &mut self,
        infer_queue_depth: usize,
        train_queue_depth: usize,
        drop_rate: f32,
    ) {
        self.telemetry.infer_queue_depth = infer_queue_depth;
        self.telemetry.train_queue_depth = train_queue_depth;
        self.telemetry.drop_rate = drop_rate.clamp(0.0, 1.0);
    }

    fn ensure_model(&mut self, slot: u32) -> Result<(), String> {
        if self.bound_models.contains_key(&slot) {
            Ok(())
        } else {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::MissingModelArtifact);
            Err(format!("mls slot {slot} has no bound model"))
        }
    }

    fn ensure_runtime_ready(&mut self, workload: MlsWorkloadKind) -> Result<(), String> {
        if self.config.mode == MlsExecutionMode::Off {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::DisabledByMode);
            return Err("mls is disabled (mode=off)".to_string());
        }
        if workload == MlsWorkloadKind::Training && !self.config.allow_training {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::TrainingDisabled);
            return Err("mls training is disabled".to_string());
        }
        if !self.config.workload_enabled(workload) {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::UnsupportedWorkload);
            return Err(format!("mls workload '{}' is disabled", workload.as_str()));
        }
        let Some(adapter) = self.active_adapter() else {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::UnsupportedBackend);
            return Err("mls has no active backend".to_string());
        };
        if workload == MlsWorkloadKind::Training && !adapter.supports_workload(workload) {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::TrainingUnsupported);
            return Err("mls training unsupported on active backend".to_string());
        }
        if !adapter.supports_workload(workload) {
            self.telemetry.fallback_reason = Some(MlsFallbackReason::UnsupportedWorkload);
            return Err(format!(
                "mls workload '{}' unsupported on active backend",
                workload.as_str()
            ));
        }
        Ok(())
    }

    fn refresh_backend(&mut self) {
        let mut fallback_reason = None;
        let adapter = if self.config.mode == MlsExecutionMode::Off {
            fallback_reason = Some(MlsFallbackReason::DisabledByMode);
            self.find_adapter(MlsBackendKind::Cpu)
        } else {
            match self.config.backend {
                MlsBackendKind::Auto => self
                    .find_first_available(&[
                        MlsBackendKind::Nvidia,
                        MlsBackendKind::Amd,
                        MlsBackendKind::Apple,
                        MlsBackendKind::Rockchip,
                        MlsBackendKind::Cpu,
                    ])
                    .or_else(|| self.find_adapter(MlsBackendKind::Cpu)),
                requested => self.find_adapter(requested).or_else(|| {
                    fallback_reason = Some(MlsFallbackReason::NativeAdapterUnavailable);
                    self.find_adapter(MlsBackendKind::Cpu)
                }),
            }
        };

        let Some(adapter) = adapter else {
            self.telemetry.backend = MlsBackendKind::Cpu;
            self.telemetry.device = "cpu".to_string();
            self.telemetry.precision = MlsPrecisionMode::Fp32;
            self.telemetry.active_workloads.clear();
            self.telemetry.fallback_reason = Some(MlsFallbackReason::UnsupportedBackend);
            return;
        };

        let profile = adapter.profile();
        let mut precision = self.config.precision;
        if precision == MlsPrecisionMode::Auto {
            precision = profile
                .capability_matrix
                .supported_precisions
                .first()
                .copied()
                .unwrap_or(MlsPrecisionMode::Fp32);
        } else if !profile
            .capability_matrix
            .supported_precisions
            .contains(&precision)
        {
            precision = profile
                .capability_matrix
                .supported_precisions
                .first()
                .copied()
                .unwrap_or(MlsPrecisionMode::Fp32);
            fallback_reason.get_or_insert(MlsFallbackReason::CpuFallback);
        }

        let mut active_workloads = Vec::new();
        for workload in [
            MlsWorkloadKind::Upscale,
            MlsWorkloadKind::Agent,
            MlsWorkloadKind::PhysicsAssist,
            MlsWorkloadKind::Training,
        ] {
            if !self.config.workload_enabled(workload) {
                continue;
            }
            if workload == MlsWorkloadKind::Training && !self.config.allow_training {
                fallback_reason.get_or_insert(MlsFallbackReason::TrainingDisabled);
                continue;
            }
            if adapter.supports_workload(workload) {
                active_workloads.push(workload);
            } else if workload == MlsWorkloadKind::Training {
                fallback_reason.get_or_insert(MlsFallbackReason::TrainingUnsupported);
            } else {
                fallback_reason.get_or_insert(MlsFallbackReason::UnsupportedWorkload);
            }
        }

        if self.config.backend == MlsBackendKind::Auto && profile.backend == MlsBackendKind::Cpu {
            fallback_reason.get_or_insert(MlsFallbackReason::CpuFallback);
        }

        self.telemetry.backend = profile.backend;
        self.telemetry.device = profile.device_name;
        self.telemetry.precision = precision;
        self.telemetry.active_workloads = active_workloads;
        self.telemetry.fallback_reason = fallback_reason;
    }

    fn active_adapter(&self) -> Option<&dyn MlsBackendAdapter> {
        self.adapters
            .iter()
            .find(|adapter| adapter.profile().backend == self.telemetry.backend)
            .map(|adapter| adapter.as_ref())
    }

    fn find_adapter(&self, kind: MlsBackendKind) -> Option<&dyn MlsBackendAdapter> {
        self.adapters
            .iter()
            .find(|adapter| adapter.kind() == kind && adapter.is_available())
            .map(|adapter| adapter.as_ref())
    }

    fn find_first_available(&self, order: &[MlsBackendKind]) -> Option<&dyn MlsBackendAdapter> {
        order.iter().find_map(|kind| self.find_adapter(*kind))
    }
}

fn default_adapters() -> Vec<Box<dyn MlsBackendAdapter>> {
    vec![
        Box::new(StaticMlsAdapter::new(
            MlsBackendKind::Amd,
            false,
            "amd-rocm",
            "gpu",
            "matrix cores",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: true,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: true,
                supported_precisions: vec![
                    MlsPrecisionMode::Fp16,
                    MlsPrecisionMode::Bf16,
                    MlsPrecisionMode::Fp32,
                    MlsPrecisionMode::Int8,
                ],
                uses_pak_model_container: true,
            },
        )),
        Box::new(StaticMlsAdapter::new(
            MlsBackendKind::Nvidia,
            false,
            "nvidia-cuda",
            "gpu",
            "tensor cores",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: true,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: true,
                supported_precisions: vec![
                    MlsPrecisionMode::Fp16,
                    MlsPrecisionMode::Bf16,
                    MlsPrecisionMode::Fp32,
                    MlsPrecisionMode::Int8,
                ],
                uses_pak_model_container: true,
            },
        )),
        Box::new(StaticMlsAdapter::new(
            MlsBackendKind::Apple,
            false,
            "apple-coreml",
            "ane",
            "neural engine",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: false,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: false,
                supported_precisions: vec![
                    MlsPrecisionMode::Fp16,
                    MlsPrecisionMode::Fp32,
                    MlsPrecisionMode::Int8,
                ],
                uses_pak_model_container: true,
            },
        )),
        Box::new(StaticMlsAdapter::new(
            MlsBackendKind::Rockchip,
            false,
            "rockchip-rknn",
            "npu",
            "npu",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: false,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: false,
                supported_precisions: vec![MlsPrecisionMode::Int8, MlsPrecisionMode::Fp16],
                uses_pak_model_container: true,
            },
        )),
        Box::new(StaticMlsAdapter::new(
            MlsBackendKind::Cpu,
            true,
            "cpu",
            "cpu",
            "scalar/simd",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: true,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: true,
                supported_precisions: vec![MlsPrecisionMode::Fp32],
                uses_pak_model_container: true,
            },
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_adapter(kind: MlsBackendKind, available: bool) -> Box<dyn MlsBackendAdapter> {
        Box::new(StaticMlsAdapter::new(
            kind,
            available,
            kind.as_str(),
            "gpu",
            "matrix",
            MlsCapabilityMatrix {
                supports_inference: true,
                supports_training: kind != MlsBackendKind::Rockchip,
                supports_upscale: true,
                supports_agent: true,
                supports_physics_assist: true,
                supports_checkpointing: true,
                supported_precisions: vec![MlsPrecisionMode::Fp16, MlsPrecisionMode::Fp32],
                uses_pak_model_container: true,
            },
        ))
    }

    #[test]
    fn auto_backend_prefers_nvidia_then_amd() {
        let runtime = MlsRuntime::with_adapters(
            MlsRuntimeConfig::default(),
            vec![
                mock_adapter(MlsBackendKind::Amd, true),
                mock_adapter(MlsBackendKind::Nvidia, true),
                mock_adapter(MlsBackendKind::Cpu, true),
            ],
        );
        assert_eq!(runtime.telemetry().backend, MlsBackendKind::Nvidia);
    }

    #[test]
    fn explicit_missing_backend_falls_back_to_cpu() {
        let mut config = MlsRuntimeConfig::default();
        config.backend = MlsBackendKind::Apple;
        let runtime =
            MlsRuntime::with_adapters(config, vec![mock_adapter(MlsBackendKind::Cpu, true)]);
        assert_eq!(runtime.telemetry().backend, MlsBackendKind::Cpu);
        assert_eq!(
            runtime.telemetry().fallback_reason,
            Some(MlsFallbackReason::NativeAdapterUnavailable)
        );
    }

    #[test]
    fn training_requires_allow_flag_and_bound_model() {
        let mut runtime = MlsRuntime::with_adapters(
            MlsRuntimeConfig::default(),
            vec![mock_adapter(MlsBackendKind::Cpu, true)],
        );
        let err = runtime.train_step(1, 4).unwrap_err();
        assert!(err.contains("no bound model"));
        runtime.bind_model(1, "models/demo.pak#agent").unwrap();
        let err = runtime.train_step(1, 4).unwrap_err();
        assert!(err.contains("training is disabled"));
    }
}
