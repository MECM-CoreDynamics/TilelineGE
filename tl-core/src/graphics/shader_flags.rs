//! Formal shader feature flags for render-pipeline variants.

use std::fmt;

/// Bitfield of active shader features that affect pipeline compilation.
///
/// These flags are part of the `PipelineKey` so that different feature
/// combinations produce different cached pipelines. Adding a flag here
/// does not automatically create a shader variant — the corresponding
/// shader source or SPIR-V artifact must also exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShaderFeatureFlags(pub u32);

impl ShaderFeatureFlags {
    pub const EMPTY: Self = Self(0);
    pub const SHADOW_ENABLED: Self = Self(1 << 0);
    pub const RT_ENABLED: Self = Self(1 << 1);
    pub const FSR_ENABLED: Self = Self(1 << 2);
    pub const SPRITE_ARRAY: Self = Self(1 << 3);

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl fmt::Display for ShaderFeatureFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (name, flag) in [
            ("shadow", Self::SHADOW_ENABLED),
            ("rt", Self::RT_ENABLED),
            ("fsr", Self::FSR_ENABLED),
            ("sprite_array", Self::SPRITE_ARRAY),
        ] {
            if self.contains(flag) {
                if !first {
                    f.write_str("|")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        if first {
            f.write_str("empty")?;
        }
        Ok(())
    }
}

impl std::ops::BitOr for ShaderFeatureFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ShaderFeatureFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
