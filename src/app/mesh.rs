use crate::app::vertex::Vertex;
use wgpu::util::DeviceExt;

/// Geometry ready to draw: its two buffers and the index count they agree on.
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    // CPU copy, so the editor can tell which mesh a click landed on.
    positions: Vec<glam::Vec3>,
    indices: Vec<u32>,
}

impl Mesh {
    pub fn new(device: &wgpu::Device, label: &str, vertices: &[Vertex], indices: &[u32]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} Vertex Buffer")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} Index Buffer")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            positions: vertices.iter().map(|v| v.position.into()).collect(),
            indices: indices.to_vec(),
        }
    }

    /// Distance along a ray (in the mesh's own space) to the nearest triangle.
    // ponytail: tests every triangle, skinned meshes in bind pose; add a bounding box if slow.
    pub fn raycast(&self, origin: glam::Vec3, direction: glam::Vec3) -> Option<f32> {
        self.indices
            .chunks_exact(3)
            .filter_map(|tri| {
                let [a, b, c] = [0, 1, 2].map(|k| self.positions[tri[k] as usize]);
                ray_triangle(origin, direction, a, b, c)
            })
            .min_by(f32::total_cmp)
    }

    /// Draws the given range of whatever instance buffer the caller bound to
    /// vertex buffer slot 1.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, instances: std::ops::Range<u32>) {
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, instances);
    }
}

/// Moller-Trumbore: how far along the ray it crosses triangle abc, if it does.
fn ray_triangle(
    origin: glam::Vec3,
    direction: glam::Vec3,
    a: glam::Vec3,
    b: glam::Vec3,
    c: glam::Vec3,
) -> Option<f32> {
    let (ab, ac) = (b - a, c - a);
    let p = direction.cross(ac);
    let det = ab.dot(p);
    if det.abs() < 1e-8 {
        return None; // parallel to the triangle
    }
    let to_origin = origin - a;
    let u = to_origin.dot(p) / det;
    let q = to_origin.cross(ab);
    let v = direction.dot(q) / det;
    let t = ac.dot(q) / det;
    (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, vec3};

    #[test]
    fn ray_hits_inside_and_misses_outside() {
        let (a, b, c) = (vec3(0.0, 0.0, 0.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 0.0, 2.0));
        let hit = ray_triangle(vec3(0.5, 3.0, 0.5), Vec3::NEG_Y, a, b, c);
        assert_eq!(hit, Some(3.0));
        assert_eq!(ray_triangle(vec3(1.5, 3.0, 1.5), Vec3::NEG_Y, a, b, c), None);
        // Behind the ray's start doesn't count.
        assert_eq!(ray_triangle(vec3(0.5, 3.0, 0.5), Vec3::Y, a, b, c), None);
    }
}
