//! Shared FBX mesh parsing for runtime scene renderers.
//!
//! Extracts position-only mesh data from FBX binary files for use by
//! both the wgpu and Vulkan scene renderer paths.

use std::io::Cursor;

use fbx::Property as FbxProperty;

/// Default embedded sphere FBX used for the high-quality sphere slot.
pub const DEFAULT_SPHERE_FBX_BYTES: &[u8] = include_bytes!("../../docs/demos/tlapp/sphere.fbx");

/// Parsed FBX mesh output: world-space positions normalized to unit box, plus triangle indices.
pub struct ParsedFbxMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Parse the first mesh geometry from an FBX binary blob.
///
/// Vertices are normalized into the `[-0.5, 0.5]` range per axis so that
/// runtime scale controls have a predictable baseline regardless of FBX authoring units.
pub fn parse_first_mesh_from_fbx(bytes: &[u8]) -> Result<ParsedFbxMesh, String> {
    let file = fbx::File::read_from(Cursor::new(bytes)).map_err(|err| format!("{err}"))?;
    let objects = file
        .children
        .iter()
        .find(|node| node.name == "Objects")
        .ok_or_else(|| "FBX objects node was not found".to_string())?;

    for geometry in objects
        .children
        .iter()
        .filter(|node| node.name == "Geometry")
    {
        let kind = geometry
            .properties
            .get(2)
            .and_then(fbx_property_as_string)
            .unwrap_or_default();
        if kind != "Mesh" {
            continue;
        }

        let vertices_f64 = geometry
            .children
            .iter()
            .find(|node| node.name == "Vertices")
            .and_then(|node| node.properties.first())
            .and_then(fbx_property_as_f64_slice)
            .ok_or_else(|| "FBX mesh does not contain vertices".to_string())?;
        let polygon_vertex_index = geometry
            .children
            .iter()
            .find(|node| node.name == "PolygonVertexIndex")
            .and_then(|node| node.properties.first())
            .and_then(fbx_property_as_i32_slice)
            .ok_or_else(|| "FBX mesh does not contain polygon vertex indices".to_string())?;

        let mut positions = parse_positions(vertices_f64)?;
        let indices = polygon_indices_to_triangles(polygon_vertex_index, positions.len())?;
        if indices.is_empty() {
            return Err("FBX mesh did not produce triangle indices".to_string());
        }
        normalize_positions_to_unit_box(&mut positions);
        return Ok(ParsedFbxMesh { positions, indices });
    }

    Err("No FBX mesh geometry node found".to_string())
}

fn parse_positions(vertices_f64: &[f64]) -> Result<Vec<[f32; 3]>, String> {
    if vertices_f64.len() < 9 {
        return Err("FBX vertices array is too small".to_string());
    }
    if vertices_f64.len() % 3 != 0 {
        return Err("FBX vertices array is not 3-component aligned".to_string());
    }
    Ok(vertices_f64
        .chunks_exact(3)
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect())
}

fn normalize_positions_to_unit_box(positions: &mut [[f32; 3]]) {
    if positions.is_empty() {
        return;
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions.iter() {
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let inv_extent = [
        if extent[0].abs() > 1e-6 {
            1.0 / extent[0]
        } else {
            0.0
        },
        if extent[1].abs() > 1e-6 {
            1.0 / extent[1]
        } else {
            0.0
        },
        if extent[2].abs() > 1e-6 {
            1.0 / extent[2]
        } else {
            0.0
        },
    ];
    for p in positions.iter_mut() {
        for axis in 0..3 {
            p[axis] = if inv_extent[axis] > 0.0 {
                (p[axis] - center[axis]) * inv_extent[axis]
            } else {
                0.0
            };
        }
    }
}

fn polygon_indices_to_triangles(
    polygon_vertex_index: &[i32],
    vertex_count: usize,
) -> Result<Vec<u32>, String> {
    let mut triangles = Vec::<u32>::new();
    let mut polygon = Vec::<u32>::with_capacity(8);

    for &raw_index in polygon_vertex_index {
        let (resolved, is_polygon_end) = if raw_index < 0 {
            let corrected = raw_index
                .checked_neg()
                .and_then(|v| v.checked_sub(1))
                .ok_or_else(|| "FBX polygon index underflow".to_string())?;
            (corrected, true)
        } else {
            (raw_index, false)
        };

        let resolved_u32: u32 = resolved
            .try_into()
            .map_err(|_| "FBX polygon index is negative".to_string())?;
        if resolved_u32 as usize >= vertex_count {
            return Err("FBX polygon index exceeds vertex count".to_string());
        }
        polygon.push(resolved_u32);

        if is_polygon_end {
            triangulate_fan(&polygon, &mut triangles);
            polygon.clear();
        }
    }

    if !polygon.is_empty() {
        triangulate_fan(&polygon, &mut triangles);
    }
    if triangles.is_empty() {
        return Err("FBX polygon list did not contain triangles".to_string());
    }
    Ok(triangles)
}

fn triangulate_fan(polygon: &[u32], out: &mut Vec<u32>) {
    if polygon.len() < 3 {
        return;
    }
    let first = polygon[0];
    for i in 1..polygon.len() - 1 {
        out.push(first);
        out.push(polygon[i]);
        out.push(polygon[i + 1]);
    }
}

fn fbx_property_as_string(property: &FbxProperty) -> Option<&str> {
    match property {
        FbxProperty::String(value) => Some(value.as_str()),
        _ => None,
    }
}

fn fbx_property_as_f64_slice(property: &FbxProperty) -> Option<&[f64]> {
    match property {
        FbxProperty::F64Array(values) => Some(values.as_slice()),
        _ => None,
    }
}

fn fbx_property_as_i32_slice(property: &FbxProperty) -> Option<&[i32]> {
    match property {
        FbxProperty::I32Array(values) => Some(values.as_slice()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_sphere_fbx_parses_into_indexed_triangles() {
        let mesh = parse_first_mesh_from_fbx(DEFAULT_SPHERE_FBX_BYTES)
            .expect("embedded sphere.fbx should parse");
        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
        let max_index = mesh.indices.iter().copied().max().unwrap_or(0) as usize;
        assert!(max_index < mesh.positions.len());
    }

    #[test]
    fn positions_normalized_to_unit_box() {
        let mesh = parse_first_mesh_from_fbx(DEFAULT_SPHERE_FBX_BYTES)
            .expect("embedded sphere.fbx should parse");
        for p in &mesh.positions {
            for &coord in p.iter() {
                assert!(
                    coord >= -0.5 - 1e-4 && coord <= 0.5 + 1e-4,
                    "position out of unit box: {coord}"
                );
            }
        }
    }
}
