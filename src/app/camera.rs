#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    matrix: [[f32; 4]; 4],
    eye: [f32; 4],
}

/// A camera that circles a point on the ground: drag to turn, scroll to zoom,
/// and move `target` to follow something or pan.
pub struct Orbit {
    pub target: glam::Vec3,
    pub yaw: f32,
    pitch: f32,
    distance: f32,
    dragging: bool,
    last_cursor: Option<glam::Vec2>,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            target: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            distance: 18.0,
            dragging: false,
            last_cursor: None,
        }
    }
}

impl Orbit {
    const RADIANS_PER_PIXEL: f32 = 0.005;
    // Top-down: never level with the ground, never quite straight down.
    const MIN_PITCH: f32 = 0.15;
    const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const MIN_DISTANCE: f32 = 3.0;
    const MAX_DISTANCE: f32 = 80.0;

    // Sperical Coordinates
    pub fn eye(&self) -> glam::Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        self.target
            + self.distance * glam::vec3(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    }

    pub fn view_proj(&self, aspect: f32) -> glam::Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye(), self.target, glam::Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(45f32.to_radians(), aspect, 0.1, 500.0);
        proj * view
    }

    /// The world-space ray under a cursor position (in pixels): origin and
    /// unit direction. This is what picking starts from.
    pub fn ray(&self, cursor: glam::Vec2, size: glam::Vec2) -> (glam::Vec3, glam::Vec3) {
        let inverse = self.view_proj(size.x / size.y).inverse();
        let ndc = glam::vec2(2.0 * cursor.x / size.x - 1.0, 1.0 - 2.0 * cursor.y / size.y);
        let near = inverse.project_point3(ndc.extend(0.0));
        let far = inverse.project_point3(ndc.extend(1.0));
        (near, (far - near).normalize())
    }

    /// Unit vectors on the ground plane pointing screen-up and screen-right,
    /// so movement keys can be camera relative.
    pub fn ground_axes(&self) -> (glam::Vec3, glam::Vec3) {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        (glam::vec3(-sin_yaw, 0.0, -cos_yaw), glam::vec3(cos_yaw, 0.0, -sin_yaw))
    }

    pub fn is_dragging(&mut self, dragging: bool) {
        self.dragging = dragging;
        self.last_cursor = None;
    }

    pub fn cursor_moved(&mut self, position: glam::Vec2) {
        let previous = self.last_cursor.replace(position);
        if !self.dragging {
            return;
        }
        let Some(previous) = previous else { return };

        let delta = position - previous;
        self.yaw -= delta.x * Self::RADIANS_PER_PIXEL;
        self.pitch = (self.pitch + delta.y * Self::RADIANS_PER_PIXEL)
            .clamp(Self::MIN_PITCH, Self::MAX_PITCH);
    }

    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance - amount).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
    }
}

/// Uniform buffer and bind group (`@group(1) @binding(0)`) for the view matrix.
pub struct Camera {
    buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Camera {
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: size_of::<CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self { buffer, bind_group }
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        eye: glam::Vec3,
    ) {
        let uniform = CameraUniform {
            matrix: view_proj.to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_ray_hits_the_target() {
        let orbit = Orbit { target: glam::vec3(3.0, 1.0, -2.0), ..Default::default() };
        let size = glam::vec2(800.0, 600.0);
        let (origin, direction) = orbit.ray(size / 2.0, size);
        let to_target = (orbit.target - origin).normalize();
        assert!(direction.dot(to_target) > 0.9999, "{direction} vs {to_target}");
    }

    #[test]
    fn ground_axes_match_the_view() {
        let orbit = Orbit { yaw: 0.7, ..Default::default() };
        let (forward, right) = orbit.ground_axes();
        let look = (orbit.target - orbit.eye()).with_y(0.0).normalize();
        assert!(forward.dot(look) > 0.9999);
        assert!(right.dot(look).abs() < 1e-4);
        // Right-handed: forward x up = right.
        assert!(forward.cross(glam::Vec3::Y).dot(right) > 0.9999);
    }
}
