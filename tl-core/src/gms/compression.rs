//! SXRC compression helpers for GMS bridge and adaptive UMA telemetry.

use crate::compression::SnapshotCompressor;
use std::time::Instant;
use sxrc::SxrcPageCodec;

pub const GMS_SXRC_DEFAULT_PAGE_SIZE: usize = 4096;
pub const GMS_SXRC_DEFAULT_MIN_BRIDGE_BYTES: usize = 64 * 1024;
pub const GMS_SXRC_DEFAULT_MAX_ENCODE_US: u64 = 500;
pub const GMS_SXRC_DEFAULT_BYPASS_FRAMES: u32 = 120;
pub const GMS_SXRC_DEFAULT_L1_PRESSURE_BYTES: usize = 32 * 1024;
pub const GMS_SXRC_DEFAULT_L2_PRESSURE_BYTES: usize = 256 * 1024;
pub const GMS_SXRC_DEFAULT_L3_PRESSURE_BYTES: usize = 2 * 1024 * 1024;

/// Runtime policy for GMS SXRC bridge/adaptive compression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmsSxrcCompressionConfig {
    pub enabled: bool,
    pub bridge_payloads: bool,
    pub adaptive_spill: bool,
    pub auto_on_cache_pressure: bool,
    pub page_size: usize,
    pub min_bridge_bytes: usize,
    pub min_savings_ratio: f64,
    pub max_encode_us_per_frame: u64,
    pub auto_bypass_frames: u32,
    pub l1_pressure_bytes: usize,
    pub l2_pressure_bytes: usize,
    pub l3_pressure_bytes: usize,
}

impl Default for GmsSxrcCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bridge_payloads: true,
            adaptive_spill: true,
            auto_on_cache_pressure: false,
            page_size: GMS_SXRC_DEFAULT_PAGE_SIZE,
            min_bridge_bytes: GMS_SXRC_DEFAULT_MIN_BRIDGE_BYTES,
            min_savings_ratio: 0.90,
            max_encode_us_per_frame: GMS_SXRC_DEFAULT_MAX_ENCODE_US,
            auto_bypass_frames: GMS_SXRC_DEFAULT_BYPASS_FRAMES,
            l1_pressure_bytes: GMS_SXRC_DEFAULT_L1_PRESSURE_BYTES,
            l2_pressure_bytes: GMS_SXRC_DEFAULT_L2_PRESSURE_BYTES,
            l3_pressure_bytes: GMS_SXRC_DEFAULT_L3_PRESSURE_BYTES,
        }
    }
}

impl GmsSxrcCompressionConfig {
    pub fn enabled_for_runtime() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    pub fn from_tileline_env() -> Self {
        let mut config = Self::default();
        let mode = std::env::var("TILELINE_SXRC")
            .unwrap_or_else(|_| "off".to_string())
            .trim()
            .to_ascii_lowercase();
        config.enabled = matches!(mode.as_str(), "gms" | "all");
        config.auto_on_cache_pressure = matches!(mode.as_str(), "auto" | "auto-gms" | "auto-all")
            || matches!(
                std::env::var("TILELINE_SXRC_AUTO")
                    .unwrap_or_else(|_| "0".to_string())
                    .trim(),
                "1" | "true" | "on" | "yes"
            );
        config
    }

    pub fn active_policy(self) -> bool {
        self.enabled || self.auto_on_cache_pressure
    }

    pub fn cache_pressure_level_for_bytes(self, bytes: usize) -> GmsSxrcCachePressureLevel {
        if bytes >= self.l3_pressure_bytes.max(1) {
            GmsSxrcCachePressureLevel::L3
        } else if bytes >= self.l2_pressure_bytes.max(1) {
            GmsSxrcCachePressureLevel::L2
        } else if bytes >= self.l1_pressure_bytes.max(1) {
            GmsSxrcCachePressureLevel::L1
        } else {
            GmsSxrcCachePressureLevel::None
        }
    }
}

/// Approximate cache level whose capacity is exceeded by a raw bridge payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GmsSxrcCachePressureLevel {
    #[default]
    None,
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GmsSxrcBypassReason {
    #[default]
    None,
    Disabled,
    BypassWindow,
    BelowMinBytes,
    RatioTooHigh,
    EncodeBudgetExceeded,
    DecodeBudgetExceeded,
    CodecError,
    CachePressureLow,
}

/// Aggregate GMS SXRC telemetry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmsSxrcCompressionStats {
    pub enabled: bool,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub sxrc_pages: u64,
    pub raw_pages: u64,
    pub encode_us: u64,
    pub decode_us: u64,
    pub stored_segments: u64,
    pub bypassed_segments: u64,
    pub auto_activated_segments: u64,
    pub bypass_frames_remaining: u32,
    pub last_bypass_reason: GmsSxrcBypassReason,
    pub cache_pressure_level: GmsSxrcCachePressureLevel,
}

impl Default for GmsSxrcCompressionStats {
    fn default() -> Self {
        Self {
            enabled: false,
            raw_bytes: 0,
            compressed_bytes: 0,
            sxrc_pages: 0,
            raw_pages: 0,
            encode_us: 0,
            decode_us: 0,
            stored_segments: 0,
            bypassed_segments: 0,
            auto_activated_segments: 0,
            bypass_frames_remaining: 0,
            last_bypass_reason: GmsSxrcBypassReason::None,
            cache_pressure_level: GmsSxrcCachePressureLevel::None,
        }
    }
}

impl GmsSxrcCompressionStats {
    pub fn compression_ratio(self) -> f64 {
        if self.raw_bytes == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.raw_bytes as f64
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GmsSxrcCompressedBlob {
    pub(crate) pages: Vec<sxrc::SxrcCompressedPage>,
    pub(crate) raw_len: usize,
    pub(crate) encoded_len: usize,
    pub(crate) sxrc_pages: usize,
    pub(crate) raw_pages: usize,
}

impl GmsSxrcCompressedBlob {
    fn from_pages(pages: Vec<sxrc::SxrcCompressedPage>) -> Self {
        let raw_len = pages.iter().map(|page| page.original_len).sum();
        let encoded_len = pages.iter().map(|page| page.encoded.len()).sum();
        let sxrc_pages = pages
            .iter()
            .filter(|page| page.codec == SxrcPageCodec::Sxrc)
            .count();
        let raw_pages = pages.len().saturating_sub(sxrc_pages);
        Self {
            pages,
            raw_len,
            encoded_len,
            sxrc_pages,
            raw_pages,
        }
    }

    pub(crate) fn ratio(&self) -> f64 {
        if self.raw_len == 0 {
            1.0
        } else {
            self.encoded_len as f64 / self.raw_len as f64
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GmsSxrcCompressor {
    config: GmsSxrcCompressionConfig,
    compressor: Option<SnapshotCompressor>,
    stats: GmsSxrcCompressionStats,
}

impl GmsSxrcCompressor {
    pub(crate) fn new(config: GmsSxrcCompressionConfig) -> Self {
        let compressor = if config.active_policy() {
            SnapshotCompressor::with_page_size(config.page_size).ok()
        } else {
            None
        };
        let mut stats = GmsSxrcCompressionStats {
            enabled: config.active_policy() && compressor.is_some(),
            ..GmsSxrcCompressionStats::default()
        };
        if config.active_policy() && compressor.is_none() {
            stats.last_bypass_reason = GmsSxrcBypassReason::CodecError;
        }
        Self {
            config,
            compressor,
            stats,
        }
    }

    pub(crate) fn stats(&self) -> GmsSxrcCompressionStats {
        self.stats
    }

    pub(crate) fn compress_bridge_payload(&mut self, data: &[u8]) -> Option<GmsSxrcCompressedBlob> {
        if !self.config.active_policy() || !self.config.bridge_payloads {
            self.record_bypass(GmsSxrcBypassReason::Disabled);
            return None;
        }
        let cache_pressure_level = self.config.cache_pressure_level_for_bytes(data.len());
        let auto_activated = !self.config.enabled
            && self.config.auto_on_cache_pressure
            && cache_pressure_level != GmsSxrcCachePressureLevel::None;
        if !self.config.enabled && !auto_activated {
            self.record_bypass(GmsSxrcBypassReason::CachePressureLow);
            return None;
        }
        if data.len() < self.config.min_bridge_bytes {
            self.record_bypass(GmsSxrcBypassReason::BelowMinBytes);
            return None;
        }
        if self.consume_bypass_frame() {
            self.record_bypass(GmsSxrcBypassReason::BypassWindow);
            return None;
        }

        let Some(compressor) = self.compressor.as_ref() else {
            self.start_bypass_window();
            self.record_bypass(GmsSxrcBypassReason::CodecError);
            return None;
        };
        let started = Instant::now();
        let Ok(pages) = compressor.compress(data) else {
            self.start_bypass_window();
            self.record_bypass(GmsSxrcBypassReason::CodecError);
            return None;
        };
        let encode_us = started.elapsed().as_micros() as u64;
        let blob = GmsSxrcCompressedBlob::from_pages(pages);

        if encode_us > self.config.max_encode_us_per_frame {
            self.start_bypass_window();
            self.record_bypass(GmsSxrcBypassReason::EncodeBudgetExceeded);
            return None;
        }
        if blob.ratio() > self.config.min_savings_ratio {
            self.record_bypass(GmsSxrcBypassReason::RatioTooHigh);
            return None;
        }

        self.stats.raw_bytes = self.stats.raw_bytes.saturating_add(blob.raw_len as u64);
        self.stats.compressed_bytes = self
            .stats
            .compressed_bytes
            .saturating_add(blob.encoded_len as u64);
        self.stats.sxrc_pages = self.stats.sxrc_pages.saturating_add(blob.sxrc_pages as u64);
        self.stats.raw_pages = self.stats.raw_pages.saturating_add(blob.raw_pages as u64);
        self.stats.encode_us = self.stats.encode_us.saturating_add(encode_us);
        self.stats.stored_segments = self.stats.stored_segments.saturating_add(1);
        if auto_activated {
            self.stats.auto_activated_segments =
                self.stats.auto_activated_segments.saturating_add(1);
        }
        self.stats.cache_pressure_level = cache_pressure_level;
        self.stats.last_bypass_reason = GmsSxrcBypassReason::None;
        Some(blob)
    }

    #[allow(dead_code)]
    pub(crate) fn decompress_bridge_payload(
        &mut self,
        blob: &GmsSxrcCompressedBlob,
    ) -> Option<Vec<u8>> {
        let Some(compressor) = self.compressor.as_ref() else {
            self.record_bypass(GmsSxrcBypassReason::CodecError);
            return None;
        };
        let started = Instant::now();
        let decoded = compressor.decompress(&blob.pages).ok()?;
        let decode_us = started.elapsed().as_micros() as u64;
        self.stats.decode_us = self.stats.decode_us.saturating_add(decode_us);
        if decode_us > self.config.max_encode_us_per_frame {
            self.start_bypass_window();
            self.record_bypass(GmsSxrcBypassReason::DecodeBudgetExceeded);
        }
        Some(decoded)
    }

    fn consume_bypass_frame(&mut self) -> bool {
        if self.stats.bypass_frames_remaining == 0 {
            return false;
        }
        self.stats.bypass_frames_remaining -= 1;
        true
    }

    fn start_bypass_window(&mut self) {
        self.stats.bypass_frames_remaining = self.config.auto_bypass_frames;
    }

    fn record_bypass(&mut self, reason: GmsSxrcBypassReason) {
        self.stats.bypassed_segments = self.stats.bypassed_segments.saturating_add(1);
        self.stats.last_bypass_reason = reason;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> GmsSxrcCompressionConfig {
        GmsSxrcCompressionConfig {
            enabled: true,
            min_bridge_bytes: 1,
            max_encode_us_per_frame: 1_000_000,
            ..GmsSxrcCompressionConfig::default()
        }
    }

    fn auto_config() -> GmsSxrcCompressionConfig {
        GmsSxrcCompressionConfig {
            enabled: false,
            auto_on_cache_pressure: true,
            min_bridge_bytes: 1,
            l1_pressure_bytes: 64,
            l2_pressure_bytes: 512,
            l3_pressure_bytes: 4096,
            max_encode_us_per_frame: 1_000_000,
            ..GmsSxrcCompressionConfig::default()
        }
    }

    #[test]
    fn bridge_payload_roundtrips_through_sxrc_blob() {
        let mut compressor = GmsSxrcCompressor::new(enabled_config());
        let bytes = vec![0u8; 8192];
        let blob = compressor
            .compress_bridge_payload(&bytes)
            .expect("zero bridge payload should compress");
        assert!(blob.encoded_len < blob.raw_len);
        let decoded = compressor.decompress_bridge_payload(&blob).expect("decode");
        assert_eq!(decoded, bytes);
        assert_eq!(compressor.stats().stored_segments, 1);
    }

    #[test]
    fn auto_mode_activates_when_bridge_payload_exceeds_cache_pressure_threshold() {
        let mut compressor = GmsSxrcCompressor::new(auto_config());
        let bytes = vec![0u8; 8192];
        let blob = compressor
            .compress_bridge_payload(&bytes)
            .expect("cache-pressured bridge payload should compress in auto mode");
        assert!(blob.encoded_len < blob.raw_len);
        let stats = compressor.stats();
        assert_eq!(stats.auto_activated_segments, 1);
        assert_eq!(stats.cache_pressure_level, GmsSxrcCachePressureLevel::L3);
    }

    #[test]
    fn auto_mode_waits_when_bridge_payload_stays_inside_cache_budget() {
        let mut compressor = GmsSxrcCompressor::new(auto_config());
        assert!(compressor.compress_bridge_payload(&[0u8; 32]).is_none());
        assert_eq!(
            compressor.stats().last_bypass_reason,
            GmsSxrcBypassReason::CachePressureLow
        );
    }

    #[test]
    fn high_entropy_bridge_payload_uses_ratio_bypass() {
        let mut compressor = GmsSxrcCompressor::new(enabled_config());
        let bytes = (0..8192)
            .map(|i| {
                let mixed = (i as u32).wrapping_mul(22_695_477).wrapping_add(1);
                (mixed >> 13) as u8
            })
            .collect::<Vec<_>>();
        assert!(compressor.compress_bridge_payload(&bytes).is_none());
        assert_eq!(
            compressor.stats().last_bypass_reason,
            GmsSxrcBypassReason::RatioTooHigh
        );
    }
}
