//! Runtime memory compression for [`RenderStateSnapshot`] data using SXRC.
//!
//! This module wraps [`sxrc::SxrcRamCodec`] and exposes a page-based
//! compressor tuned for engine buffer payloads (instance transforms,
//! light records, etc.).  When compression does not yield savings the
//! codec falls back to raw storage automatically.

use sxrc::{SxrcCodecConfig, SxrcCompressedPage, SxrcManifest, SxrcRamCodec, SxrcStats};

/// Default 4 KiB page size used for frame-slot buffer compression.
const DEFAULT_PAGE_SIZE: usize = 4096;

/// Lightweight wrapper around [`SxrcRamCodec`] with a built-in
/// generic manifest suitable for engine buffer data.
#[derive(Debug, Clone)]
pub struct SnapshotCompressor {
    codec: SxrcRamCodec,
    page_size: usize,
}

impl Default for SnapshotCompressor {
    fn default() -> Self {
        Self::new().expect("default manifest is valid")
    }
}

impl SnapshotCompressor {
    /// Create a compressor with the built-in generic manifest.
    pub fn new() -> Result<Self, sxrc::SxrcError> {
        let manifest = generic_engine_manifest();
        let config = SxrcCodecConfig {
            page_size: DEFAULT_PAGE_SIZE,
            ..SxrcCodecConfig::from_manifest(&manifest)
        };
        let codec = SxrcRamCodec::new(&manifest, config)?;
        Ok(Self {
            codec,
            page_size: DEFAULT_PAGE_SIZE,
        })
    }

    /// Create a compressor with a custom manifest and config.
    pub fn with_manifest(
        manifest: &SxrcManifest,
        config: SxrcCodecConfig,
    ) -> Result<Self, sxrc::SxrcError> {
        let codec = SxrcRamCodec::new(manifest, config)?;
        Ok(Self {
            codec,
            page_size: config.page_size,
        })
    }

    /// Compress a contiguous byte slice into a vector of SXRC pages.
    ///
    /// The slice is split into `page_size`-sized chunks.  Each chunk
    /// is compressed independently; pages that do not shrink are stored
    /// as [`SxrcPageCodec::Raw`] fallback.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<SxrcCompressedPage>, sxrc::SxrcError> {
        let mut pages = Vec::with_capacity(data.len().div_ceil(self.page_size));
        for chunk in data.chunks(self.page_size) {
            pages.push(self.codec.compress_page(chunk)?);
        }
        Ok(pages)
    }

    /// Decompress a vector of pages back into a contiguous byte vector.
    pub fn decompress(&self, pages: &[SxrcCompressedPage]) -> Result<Vec<u8>, sxrc::SxrcError> {
        let mut out = Vec::with_capacity(
            pages.iter().map(|p| p.original_len).sum(),
        );
        for page in pages {
            out.extend_from_slice(&self.codec.decompress_page(page)?);
        }
        Ok(out)
    }

    /// Return the aggregate compression ratio for a set of pages.
    ///
    /// A ratio of `0.5` means the data occupies half the original size.
    pub fn ratio(pages: &[SxrcCompressedPage]) -> f64 {
        let raw: usize = pages.iter().map(|p| p.original_len).sum();
        let encoded: usize = pages.iter().map(|p| p.encoded.len()).sum();
        if raw == 0 {
            1.0
        } else {
            encoded as f64 / raw as f64
        }
    }

    /// Aggregate [`SxrcStats`] across a set of pages.
    pub fn stats(pages: &[SxrcCompressedPage]) -> SxrcStats {
        let mut total = SxrcStats::default();
        for page in pages {
            total.raw_bytes += page.stats.raw_bytes;
            total.encoded_bytes += page.stats.encoded_bytes;
            total.token_count += page.stats.token_count;
            total.literal_tokens += page.stats.literal_tokens;
            total.dict_tokens += page.stats.dict_tokens;
            total.pattern_tokens += page.stats.pattern_tokens;
            total.rle_tokens += page.stats.rle_tokens;
            total.raw_escape_tokens += page.stats.raw_escape_tokens;
            total.dynamic_pattern_count += page.stats.dynamic_pattern_count;
            total.dynamic_metadata_bytes += page.stats.dynamic_metadata_bytes;
        }
        total
    }
}

/// Build a minimal generic SXRC manifest for engine buffer data.
///
/// The manifest is intentionally sparse — it enables the SXRC encoder
/// to discover dynamic patterns and RLE runs without prescribing
/// architecture-specific dictionary entries.
fn generic_engine_manifest() -> SxrcManifest {
    SxrcManifest {
        version: "0.1.0".to_string(),
        target_arch: "generic".to_string(),
        compression_unit: sxrc::CompressionUnit::U16,
        endian: sxrc::Endian::Little,
        static_dictionary: vec![],
        instruction_patterns: vec![],
        memory_markers: std::collections::BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zeroes() {
        let compressor = SnapshotCompressor::new().unwrap();
        let data = vec![0u8; 8192];
        let pages = compressor.compress(&data).unwrap();
        let decoded = compressor.decompress(&pages).unwrap();
        assert_eq!(data, decoded);
        let ratio = SnapshotCompressor::ratio(&pages);
        assert!(ratio < 0.1, "zero-filled data should compress well, got ratio {ratio}");
    }

    #[test]
    fn roundtrip_random() {
        let compressor = SnapshotCompressor::new().unwrap();
        let data: Vec<u8> = (0..4096).map(|i| (i * 7 + 13) as u8).collect();
        let pages = compressor.compress(&data).unwrap();
        let decoded = compressor.decompress(&pages).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn ratio_bounds() {
        let compressor = SnapshotCompressor::new().unwrap();
        let data = vec![0xABu8; 4096];
        let pages = compressor.compress(&data).unwrap();
        let r = SnapshotCompressor::ratio(&pages);
        assert!(r >= 0.0 && r <= 1.0, "ratio should be in [0,1], got {r}");
    }
}
