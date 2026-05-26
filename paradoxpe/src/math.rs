//! Mathematical extensions and safe wrappers for physics robustness.

pub trait SafeClamp {
    /// Clamps a value between `min` and `max`.
    /// 
    /// Unlike `f32::clamp`, this method safely handles `NaN` values:
    /// - If `self` is `NaN`, it defaults to `0.0`.
    /// - If `min` or `max` is `NaN`, they default to `0.0`.
    /// - If `min > max` (possibly after falling back to `0.0`), they are safely swapped to prevent panic.
    fn safe_clamp(self, min: Self, max: Self) -> Self;
}

impl SafeClamp for f32 {
    #[inline(always)]
    fn safe_clamp(mut self, mut min: f32, mut max: f32) -> f32 {
        if self.is_nan() {
            self = 0.0;
        }
        if min.is_nan() {
            min = 0.0;
        }
        if max.is_nan() {
            max = 0.0;
        }
        if min > max {
            std::mem::swap(&mut min, &mut max);
        }
        self.clamp(min, max)
    }
}
