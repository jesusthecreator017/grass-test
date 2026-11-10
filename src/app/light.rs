use crate::app::gpu::DEPTH_FORMAT;

pub const SHADOW_SIZE: u32 = 2048;
/// Half the side of the square of world the shadow map covers.
// ponytail: one fixed box; use cascades for more reach or sharpness.
const SHADOW_EXTENT: f32 = 40.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    direction: [f32; 4],
    color: [f32; 4],
    /// x: ambient, y: specular strength, z: extra shadow depth bias.
    params: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

/// A directional light and the shadow map it casts.
pub struct Light {
    pub direction: glam::Vec3,
    pub color: glam::Vec3,
    pub ambient: f32,
    pub specular: f32,
    pub shadow_bias: f32,
    buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub shadow_view: wgpu::TextureView,
}

impl Light {
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Bind Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        })
    }

    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        direction: glam::Vec3,
        color: glam::Vec3,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Buffer"),
            size: size_of::<LightUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Shadow Map"),
                size: wgpu::Extent3d {
                    width: SHADOW_SIZE,
                    height: SHADOW_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            direction: direction.normalize(),
            color,
            ambient: 0.3,
            specular: 0.15,
            shadow_bias: 0.0,
            buffer,
            bind_group,
            shadow_view,
        }
    }

    /// The matrix the shadow pass renders with: an orthographic box looking
    /// along the light, centred on `focus`.
    pub fn view_proj(&self, focus: glam::Vec3) -> glam::Mat4 {
        let e = SHADOW_EXTENT;
        let eye = focus - self.direction * (2.0 * e);
        let view = glam::camera::rh::view::look_at_mat4(eye, focus, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::orthographic(-e, e, -e, e, 0.1, 4.0 * e);
        proj * view
    }

    pub fn upload(&self, queue: &wgpu::Queue, focus: glam::Vec3) {
        let uniform = LightUniform {
            direction: self.direction.extend(0.0).to_array(),
            color: self.color.extend(1.0).to_array(),
            params: [self.ambient, self.specular, self.shadow_bias, 0.0],
            view_proj: self.view_proj(focus).to_cols_array_2d(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
    }
}
