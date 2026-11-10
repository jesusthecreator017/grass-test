#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_coord: [f32; 2],
    pub normal: [f32; 3],
    /// Skinning data. All-zero weights mean "not skinned": the shader then
    /// uses the instance matrix alone.
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}       

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Uint32x4, 4 => Float32x4
    ];

    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

}
    
pub const CUBE: [Vertex; 24] = [
    // Front (Z+)
    Vertex { position: [-0.5, -0.5, 0.5], uv_coord: [0.0, 1.0], normal: [0.0, 0.0, 1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5, -0.5, 0.5], uv_coord: [1.0, 1.0], normal: [0.0, 0.0, 1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5, 0.5], uv_coord: [1.0, 0.0], normal: [0.0, 0.0, 1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5,  0.5, 0.5], uv_coord: [0.0, 0.0], normal: [0.0, 0.0, 1.0], joints: [0; 4], weights: [0.0; 4] },

    // Back (Z-)
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coord: [0.0, 1.0], normal: [0.0, 0.0, -1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5, -0.5, -0.5], uv_coord: [1.0, 1.0], normal: [0.0, 0.0, -1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5,  0.5, -0.5], uv_coord: [1.0, 0.0], normal: [0.0, 0.0, -1.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coord: [0.0, 0.0], normal: [0.0, 0.0, -1.0], joints: [0; 4], weights: [0.0; 4] },

    // right (+x)
    Vertex { position: [ 0.5, -0.5,  0.5], uv_coord: [0.0, 1.0], normal: [1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coord: [1.0, 1.0], normal: [1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coord: [1.0, 0.0], normal: [1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5,  0.5], uv_coord: [0.0, 0.0], normal: [1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },

    // left (-x)
    Vertex { position: [-0.5, -0.5, -0.5], uv_coord: [0.0, 1.0], normal: [-1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5, -0.5,  0.5], uv_coord: [1.0, 1.0], normal: [-1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5,  0.5,  0.5], uv_coord: [1.0, 0.0], normal: [-1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5,  0.5, -0.5], uv_coord: [0.0, 0.0], normal: [-1.0, 0.0, 0.0], joints: [0; 4], weights: [0.0; 4] },

    // top (+y)
    Vertex { position: [-0.5,  0.5,  0.5], uv_coord: [0.0, 1.0], normal: [0.0, 1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5,  0.5], uv_coord: [1.0, 1.0], normal: [0.0, 1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coord: [1.0, 0.0], normal: [0.0, 1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5,  0.5, -0.5], uv_coord: [0.0, 0.0], normal: [0.0, 1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },

    // bottom (-y)
    Vertex { position: [-0.5, -0.5, -0.5], uv_coord: [0.0, 1.0], normal: [0.0, -1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coord: [1.0, 1.0], normal: [0.0, -1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [ 0.5, -0.5,  0.5], uv_coord: [1.0, 0.0], normal: [0.0, -1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
    Vertex { position: [-0.5, -0.5,  0.5], uv_coord: [0.0, 0.0], normal: [0.0, -1.0, 0.0], joints: [0; 4], weights: [0.0; 4] },
];

pub const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
];

#[cfg(test)]
mod tests {
    use super::*;

    // Every face must wind counter-clockwise seen from outside, or back-face
    // culling hides the wrong half of the cube.
    #[test]
    fn faces_wind_outward() {
        for tri in CUBE_INDICES.chunks(3) {
            let p: Vec<[f32; 3]> = tri.iter().map(|&i| CUBE[i as usize].position).collect();
            let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            // The cube is centred on the origin, so a triangle's own centre
            // doubles as the direction it should face.
            let center = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
                (p[0][2] + p[1][2] + p[2][2]) / 3.0,
            ];
            let facing = dot(normal, center);
            assert!(facing > 0.0, "triangle {tri:?} winds inward (dot = {facing})");
        }
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}

#[cfg(test)]
mod normal_tests {
    use super::*;

    /// The stored normal must agree with the winding, or lighting and culling
    /// disagree about which way a face points.
    #[test]
    fn stored_normals_match_geometry() {
        for tri in CUBE_INDICES.chunks(3) {
            let v: Vec<Vertex> = tri.iter().map(|&i| CUBE[i as usize]).collect();
            let geometric = normalize(cross(
                sub(v[1].position, v[0].position),
                sub(v[2].position, v[0].position),
            ));

            for vertex in &v {
                let agreement = dot(geometric, normalize(vertex.normal));
                assert!(
                    agreement > 0.999,
                    "triangle {tri:?}: stored {:?} disagrees with geometry {geometric:?}",
                    vertex.normal
                );
            }
        }
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn normalize(a: [f32; 3]) -> [f32; 3] {
        let len = dot(a, a).sqrt();
        [a[0] / len, a[1] / len, a[2] / len]
    }
}
