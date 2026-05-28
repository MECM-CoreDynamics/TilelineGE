//! CPU-side culling helpers for the raw Vulkan renderer.
//!
//! This stays separate from `vulkan_backend` so visibility policy can be tested and evolved
//! without threading more state through the swapchain/pipeline code.

#![cfg(target_os = "linux")]

use crate::graphics::frame_snapshot::{
    FrameInstanceTransform, FramePrimitiveRange, RenderStateSnapshot,
    FRAME_PRIMITIVE_RANGE_OVERLAY, FRAME_PRIMITIVE_RANGE_TRANSPARENT,
};

#[inline]
fn primitive_radius_sq(primitive_code: u32) -> f32 {
    match primitive_code {
        0 => 1.0,  // unit sphere
        1 => 0.75, // (sqrt(3)/2)^2: circumscribed sphere of unit box
        _ => 1.0,  // custom mesh: conservative
    }
}

/// Extract six view-frustum planes from a column-major view-projection matrix.
/// Planes are normalized (xyz) and oriented so that `dot(point, xyz) + w >= 0`
/// means the point is on the inside side.
#[inline]
pub(crate) fn extract_frustum_planes(view_proj: [[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let mut p = [[0.0f32; 4]; 6];
    // column-major layout: view_proj[col][row] = M[row][col]
    // row i = [view_proj[0][i], view_proj[1][i], view_proj[2][i], view_proj[3][i]]
    for c in 0..4 {
        p[0][c] = view_proj[c][0] + view_proj[c][3]; // left   = row0 + row3
        p[1][c] = view_proj[c][3] - view_proj[c][0]; // right  = row3 - row0
        p[2][c] = view_proj[c][1] + view_proj[c][3]; // bottom = row1 + row3
        p[3][c] = view_proj[c][3] - view_proj[c][1]; // top    = row3 - row1
        p[4][c] = view_proj[c][2] + view_proj[c][3]; // near   = row2 + row3 (Vulkan z in [0,w])
        p[5][c] = view_proj[c][3] - view_proj[c][2]; // far    = row3 - row2
    }
    for plane in &mut p {
        let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])
            .sqrt()
            .max(1e-6);
        plane[0] /= len;
        plane[1] /= len;
        plane[2] /= len;
        plane[3] /= len;
    }
    p
}

#[inline]
fn is_sphere_visible(center: [f32; 3], radius_sq: f32, planes: &[[f32; 4]; 6]) -> bool {
    for plane in planes {
        let dist = plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3];
        if dist < 0.0 && dist * dist > radius_sq {
            return false;
        }
    }
    true
}

#[inline]
fn is_transparent_range(range: &FramePrimitiveRange) -> bool {
    range.flags & FRAME_PRIMITIVE_RANGE_TRANSPARENT != 0
}

#[inline]
fn is_overlay_range(range: &FramePrimitiveRange) -> bool {
    range.flags & FRAME_PRIMITIVE_RANGE_OVERLAY != 0
}

/// CPU-side frustum cull + compaction.
///
/// Returns a compacted instance buffer containing only visible instances,
/// and rewritten primitive ranges that point into the compacted buffer.
pub(crate) fn cull_and_compact(
    snapshot: &RenderStateSnapshot<'_>,
    planes: &[[f32; 4]; 6],
) -> (Vec<FrameInstanceTransform>, Vec<FramePrimitiveRange>) {
    let mut visible = Vec::with_capacity(snapshot.transforms.len());
    let mut ranges = Vec::with_capacity(snapshot.primitive_ranges.len());

    for range in snapshot.primitive_ranges {
        // Transparent and overlay ranges are never frustum-culled.
        if is_transparent_range(range) || is_overlay_range(range) {
            let visible_start = visible.len() as u32;
            for i in range.first_instance..range.first_instance + range.instance_count {
                if let Some(t) = snapshot.transforms.get(i as usize) {
                    visible.push(*t);
                }
            }
            if range.instance_count > 0 {
                ranges.push(FramePrimitiveRange {
                    primitive_code: range.primitive_code,
                    first_instance: visible_start,
                    instance_count: range.instance_count,
                    flags: range.flags,
                });
            }
            continue;
        }

        let base_radius_sq = primitive_radius_sq(range.primitive_code);
        let start = range.first_instance;
        let end = range.first_instance + range.instance_count;
        let mut block_start: Option<u32> = None;

        for i in start..end {
            let inst = match snapshot.transforms.get(i as usize) {
                Some(t) => t,
                None => continue,
            };
            let center = [inst.model[3][0], inst.model[3][1], inst.model[3][2]];
            let sx = inst.model[0][0] * inst.model[0][0]
                + inst.model[0][1] * inst.model[0][1]
                + inst.model[0][2] * inst.model[0][2];
            let sy = inst.model[1][0] * inst.model[1][0]
                + inst.model[1][1] * inst.model[1][1]
                + inst.model[1][2] * inst.model[1][2];
            let sz = inst.model[2][0] * inst.model[2][0]
                + inst.model[2][1] * inst.model[2][1]
                + inst.model[2][2] * inst.model[2][2];
            let radius_sq = base_radius_sq * sx.max(sy).max(sz);

            if is_sphere_visible(center, radius_sq, planes) {
                visible.push(*inst);
                if block_start.is_none() {
                    block_start = Some((visible.len() - 1) as u32);
                }
            } else if let Some(bs) = block_start {
                ranges.push(FramePrimitiveRange {
                    primitive_code: range.primitive_code,
                    first_instance: bs,
                    instance_count: (visible.len() as u32) - bs,
                    flags: range.flags,
                });
                block_start = None;
            }
        }

        if let Some(bs) = block_start {
            ranges.push(FramePrimitiveRange {
                primitive_code: range.primitive_code,
                first_instance: bs,
                instance_count: (visible.len() as u32) - bs,
                flags: range.flags,
            });
        }
    }

    merge_ranges(&mut ranges);
    (visible, ranges)
}

/// Merge consecutive primitive ranges that share the same mesh and contiguous instances.
fn merge_ranges(ranges: &mut Vec<FramePrimitiveRange>) {
    if ranges.len() < 2 {
        return;
    }
    let mut write = 0;
    for read in 1..ranges.len() {
        let prev = ranges[write];
        let curr = ranges[read];
        if prev.primitive_code == curr.primitive_code
            && prev.first_instance + prev.instance_count == curr.first_instance
            && prev.flags == curr.flags
        {
            ranges[write].instance_count += curr.instance_count;
        } else {
            write += 1;
            ranges[write] = curr;
        }
    }
    ranges.truncate(write + 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::frame_snapshot::FRAME_PRIMITIVE_RANGE_OVERLAY;

    fn transform_at(x: f32, y: f32, z: f32) -> FrameInstanceTransform {
        let mut model = [[0.0; 4]; 4];
        model[0][0] = 1.0;
        model[1][1] = 1.0;
        model[2][2] = 1.0;
        model[3][3] = 1.0;
        model[3][0] = x;
        model[3][1] = y;
        model[3][2] = z;
        FrameInstanceTransform {
            model,
            ..FrameInstanceTransform::default()
        }
    }

    fn snapshot<'a>(
        transforms: &'a [FrameInstanceTransform],
        ranges: &'a [FramePrimitiveRange],
    ) -> RenderStateSnapshot<'a> {
        RenderStateSnapshot {
            frame_id: 1,
            camera_view_proj: [[0.0; 4]; 4],
            camera_eye: [0.0; 4],
            opaque_instance_count: transforms.len() as u32,
            transparent_instance_count: 0,
            primitive_ranges: ranges,
            transforms,
            materials: &[],
            textures: &[],
            lights: &[],
            sprites: &[],
            instance_bounds: &[],
            vertices: &[],
            indices: &[],
        }
    }

    #[test]
    fn overlay_ranges_are_preserved_by_compaction() {
        let transforms = [transform_at(1_000.0, 0.0, 0.0)];
        let ranges = [FramePrimitiveRange {
            primitive_code: 1,
            first_instance: 0,
            instance_count: 1,
            flags: FRAME_PRIMITIVE_RANGE_OVERLAY,
        }];
        let snapshot = snapshot(&transforms, &ranges);
        let planes = [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ];

        let (visible, compacted) = cull_and_compact(&snapshot, &planes);
        assert_eq!(visible.len(), 1);
        assert_eq!(compacted, ranges);
    }

    #[test]
    fn opaque_ranges_can_be_culled_and_split() {
        let transforms = [
            transform_at(0.0, 0.0, 0.0),
            transform_at(4.0, 0.0, 0.0),
            transform_at(0.0, 0.0, 0.0),
        ];
        let ranges = [FramePrimitiveRange {
            primitive_code: 0,
            first_instance: 0,
            instance_count: 3,
            flags: 0,
        }];
        let snapshot = snapshot(&transforms, &ranges);
        let planes = [
            [1.0, 0.0, 0.0, 2.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, -1.0, 2.0],
        ];

        let (visible, compacted) = cull_and_compact(&snapshot, &planes);
        assert_eq!(visible.len(), 2);
        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].first_instance, 0);
        assert_eq!(compacted[0].instance_count, 2);
    }
}
