#![cfg(target_os = "linux")]

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

pub(crate) fn unit_cube_vertices() -> [SceneVertex; 8] {
    [
        SceneVertex {
            position: [-0.5, -0.5, -0.5],
            uv: [0.0, 0.0],
        },
        SceneVertex {
            position: [0.5, -0.5, -0.5],
            uv: [1.0, 0.0],
        },
        SceneVertex {
            position: [0.5, 0.5, -0.5],
            uv: [1.0, 1.0],
        },
        SceneVertex {
            position: [-0.5, 0.5, -0.5],
            uv: [0.0, 1.0],
        },
        SceneVertex {
            position: [-0.5, -0.5, 0.5],
            uv: [0.0, 0.0],
        },
        SceneVertex {
            position: [0.5, -0.5, 0.5],
            uv: [1.0, 0.0],
        },
        SceneVertex {
            position: [0.5, 0.5, 0.5],
            uv: [1.0, 1.0],
        },
        SceneVertex {
            position: [-0.5, 0.5, 0.5],
            uv: [0.0, 1.0],
        },
    ]
}

pub(crate) fn unit_cube_indices() -> [u16; 36] {
    [
        0, 1, 2, 2, 3, 0, // back
        4, 5, 6, 6, 7, 4, // front
        0, 4, 7, 7, 3, 0, // left
        1, 5, 6, 6, 2, 1, // right
        3, 2, 6, 6, 7, 3, // top
        0, 1, 5, 5, 4, 0, // bottom
    ]
}

pub(crate) fn unit_icosa_sphere_vertices() -> [SceneVertex; 12] {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut vertices = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    for position in &mut vertices {
        let length =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt()
                .max(1e-6);
        position[0] /= length;
        position[1] /= length;
        position[2] /= length;
    }
    vertices.map(|position| {
        let u = 0.5 + f32::atan2(position[2], position[0]) / (2.0 * std::f32::consts::PI);
        let v = 0.5 - position[1];
        SceneVertex {
            position,
            uv: [u, v],
        }
    })
}

pub(crate) fn unit_icosa_sphere_indices() -> [u16; 60] {
    [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube_geometry_counts_match_indexed_box() {
        assert_eq!(unit_cube_vertices().len(), 8);
        assert_eq!(unit_cube_indices().len(), 36);
    }

    #[test]
    fn unit_icosa_sphere_vertices_are_normalized() {
        for vertex in unit_icosa_sphere_vertices() {
            let [x, y, z] = vertex.position;
            let length = (x * x + y * y + z * z).sqrt();
            assert!((length - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn unit_icosa_sphere_geometry_counts_match_icosahedron() {
        assert_eq!(unit_icosa_sphere_vertices().len(), 12);
        assert_eq!(unit_icosa_sphere_indices().len(), 60);
    }
}
