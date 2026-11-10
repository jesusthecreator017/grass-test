//! Props: scenery drawn as a few meshes instanced thousands of times (not
//! entities), plus the tools to build those meshes in code.

use crate::app::gpu::{DEPTH_FORMAT, Gpu, SAMPLE_COUNT};
use crate::app::renderer::{ShaderFile, try_build};
use crate::app::terrain;
use glam::{Mat4, Quat, Vec3, vec3};
use std::f32::consts::{PI, TAU};
use wgpu::util::DeviceExt;

/// Props are grouped into square chunks this many metres a side, and drawn
/// a chunk at a time, so only the ones near the view cost anything.
const CHUNK: f32 = 12.5;
/// Radius of a sphere sure to hold a chunk and everything standing in it.
const CHUNK_RADIUS: f32 = 13.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PropVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub sway: f32,
}

/// One copy of a prop.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PropInstance {
    pub position: [f32; 3],
    pub scale: f32,
    /// Rotation about the vertical, in radians.
    pub yaw: f32,
    pub tint: [f32; 3],
}

/// Small repeatable random numbers: the same seed grows the same field.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453) | 1)
    }

    /// The next number in 0..1.
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0
    }

    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[((self.next() * items.len() as f32) as usize).min(items.len() - 1)]
    }

    /// A random direction.
    pub fn direction(&mut self) -> Vec3 {
        let y = self.range(-1.0, 1.0);
        let around = self.range(0.0, TAU);
        let flat = (1.0 - y * y).sqrt();
        vec3(flat * around.cos(), y, flat * around.sin())
    }
}

/// The outline of a ribbon: how wide it is along its length.
#[derive(Clone, Copy)]
pub enum Shape {
    /// Widest a third of the way up, pointed at both ends.
    Leaf,
    /// Long and narrow, tapering from the base.
    Sword,
    /// Narrow at the base, wide and rounded towards the tip.
    Petal,
}

impl Shape {
    fn width(self, t: f32) -> f32 {
        match self {
            Shape::Leaf => (PI * t.powf(0.6)).sin().max(0.0),
            Shape::Sword => (1.0 - t).powf(0.6) * (t * 6.0).clamp(0.25, 1.0),
            Shape::Petal => (PI * t.powf(1.6)).sin().max(0.0).sqrt(),
        }
    }
}

/// A leaf or petal strip from `base`, heading `out`, rising `lift` radians and
/// turning `curl` along its length (negative droops).
#[derive(Clone, Copy)]
pub struct Ribbon {
    pub base: Vec3,
    pub out: Vec3,
    pub length: f32,
    pub width: f32,
    pub lift: f32,
    pub curl: f32,
    /// How far the edges fold up around the middle, making a V or a cup.
    pub fold: f32,
    pub shape: Shape,
    /// Colour at the base and at the tip.
    pub c0: Vec3,
    pub c1: Vec3,
    /// Colour of the edges, if different.
    pub rim: Option<Vec3>,
    pub segments: usize,
}

impl Default for Ribbon {
    fn default() -> Self {
        Self {
            base: Vec3::ZERO,
            out: Vec3::X,
            length: 0.2,
            width: 0.05,
            lift: 1.0,
            curl: 0.0,
            fold: 0.3,
            shape: Shape::Leaf,
            c0: vec3(0.03, 0.16, 0.05),
            c1: vec3(0.12, 0.38, 0.08),
            rim: None,
            segments: 5,
        }
    }
}

/// A transform that puts the origin at `origin` and points +Y along `direction`.
pub fn facing(origin: Vec3, direction: Vec3) -> Mat4 {
    Mat4::from_translation(origin) * Mat4::from_quat(Quat::from_rotation_arc(Vec3::Y, direction))
}

/// Builds a prop's mesh out of simple parts. Winding doesn't matter: props
/// are drawn two-sided and lit on whichever side faces the camera.
#[derive(Default)]
pub struct MeshBuilder {
    pub vertices: Vec<PropVertex>,
    pub indices: Vec<u32>,
    transform: Mat4,
}

impl MeshBuilder {
    /// Runs `build` with everything it adds moved by `transform`.
    pub fn at(&mut self, transform: Mat4, build: impl FnOnce(&mut Self)) {
        let outer = self.transform;
        self.transform = outer * transform;
        build(self);
        self.transform = outer;
    }

    pub fn push(&mut self, position: Vec3, normal: Vec3, color: Vec3) -> u32 {
        self.vertices.push(PropVertex {
            position: self.transform.transform_point3(position).to_array(),
            normal: self.transform.transform_vector3(normal).normalize_or(Vec3::Y).to_array(),
            color: color.to_array(),
            sway: 0.0,
        });
        self.vertices.len() as u32 - 1
    }

    pub fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Returns where the ribbon ends and which way it was heading there.
    pub fn ribbon(&mut self, r: &Ribbon) -> (Vec3, Vec3) {
        let side = r.out.cross(Vec3::Y).normalize_or(Vec3::Z);
        let step = r.length / r.segments as f32;
        let mut position = r.base;
        let mut tangent = Vec3::Y;
        let mut rows: Vec<[u32; 3]> = Vec::with_capacity(r.segments + 1);

        for i in 0..=r.segments {
            let t = i as f32 / r.segments as f32;
            let angle = r.lift + r.curl * t;
            tangent = r.out * angle.cos() + Vec3::Y * angle.sin();
            let normal = side.cross(tangent);
            let half = r.width * 0.5 * r.shape.width(t);
            let raised = normal * (r.fold * half);
            let middle = r.c0.lerp(r.c1, t);
            let edge = r.rim.unwrap_or(middle * 1.12);

            rows.push([
                self.push(position - side * half + raised, normal + side * r.fold, edge),
                self.push(position, normal, middle),
                self.push(position + side * half + raised, normal - side * r.fold, edge),
            ]);
            if i < r.segments {
                position += tangent * step;
            }
        }

        for pair in rows.windows(2) {
            let ([a0, a1, a2], [b0, b1, b2]) = (pair[0], pair[1]);
            self.triangle(a0, a1, b1);
            self.triangle(a0, b1, b0);
            self.triangle(a1, a2, b2);
            self.triangle(a1, b2, b1);
        }
        (position, tangent)
    }

    /// An open tube from `a` to `b`: a stem, or a flower's trumpet.
    #[allow(clippy::too_many_arguments)]
    pub fn tube(&mut self, a: Vec3, b: Vec3, r0: f32, r1: f32, c0: Vec3, c1: Vec3, sides: u32) {
        let axis = (b - a).normalize_or(Vec3::Y);
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        let first = self.vertices.len() as u32;
        for (centre, radius, color) in [(a, r0, c0), (b, r1, c1)] {
            for i in 0..sides {
                let angle = TAU * i as f32 / sides as f32;
                let normal = u * angle.cos() + v * angle.sin();
                self.push(centre + normal * radius, normal, color);
            }
        }
        for i in 0..sides {
            let j = (i + 1) % sides;
            self.triangle(first + i, first + j, first + sides + j);
            self.triangle(first + i, first + sides + j, first + sides + i);
        }
    }

    /// A stem from `from` to `to`, bowed sideways by `bow` in the middle.
    /// Returns the direction it points at the top.
    pub fn stalk(&mut self, from: Vec3, to: Vec3, bow: Vec3, radius: f32, color: Vec3) -> Vec3 {
        const PIECES: usize = 4;
        let control = (from + to) * 0.5 + bow;
        let point = |t: f32| from.lerp(control, t).lerp(control.lerp(to, t), t);
        let thickness = |t: f32| radius * (1.0 - 0.4 * t);
        for i in 0..PIECES {
            let (t0, t1) = (i as f32 / PIECES as f32, (i + 1) as f32 / PIECES as f32);
            self.tube(point(t0), point(t1), thickness(t0), thickness(t1), color, color, 4);
        }
        (to - control).normalize_or(Vec3::Y)
    }

    /// A rough ball, coloured `low` underneath to `high` on top.
    pub fn blob(&mut self, centre: Vec3, radii: Vec3, low: Vec3, high: Vec3) {
        const ROWS: u32 = 4;
        const COLUMNS: u32 = 6;
        let first = self.vertices.len() as u32;
        for row in 0..=ROWS {
            let down = PI * row as f32 / ROWS as f32;
            for column in 0..=COLUMNS {
                let around = TAU * column as f32 / COLUMNS as f32;
                let n = vec3(down.sin() * around.cos(), down.cos(), down.sin() * around.sin());
                self.push(centre + n * radii, n / radii, low.lerp(high, n.y * 0.5 + 0.5));
            }
        }
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let a = first + row * (COLUMNS + 1) + column;
                let b = a + COLUMNS + 1;
                self.triangle(a, b, a + 1);
                self.triangle(a + 1, b, b + 1);
            }
        }
    }

    /// Makes the mesh bend in the wind: not at all at the ground, by `flex`
    /// at its top.
    pub fn with_sway(mut self, flex: f32) -> Self {
        let top = self.vertices.iter().map(|v| v.position[1]).fold(0.01, f32::max);
        for vertex in &mut self.vertices {
            vertex.sway = (vertex.position[1] / top).clamp(0.0, 1.0).powf(1.5) * flex;
        }
        self
    }
}

/// Can any part of a sphere be on screen?
pub fn sphere_visible(view_proj: Mat4, centre: Vec3, radius: f32) -> bool {
    let (x, y, z, w) = (view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3));
    // The six sides of the view volume; depth runs 0..1.
    [w + x, w - x, w + y, w - y, z, w - z].iter().all(|plane| {
        let normal = plane.truncate();
        normal.dot(centre) + plane.w >= -radius * normal.length()
    })
}

struct Chunk {
    cell: (u32, u32),
    centre: Vec3,
    instances: std::ops::Range<u32>,
}

/// One mesh and where its copies are, chunk by chunk.
struct Kind {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    chunks: Vec<Chunk>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AirUniform {
    wind: [f32; 4],
    time: [f32; 4],
}

pub struct Props {
    shader: ShaderFile,
    air_layout: wgpu::BindGroupLayout,
    air: wgpu::Buffer,
    air_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    kinds: Vec<Kind>,
    instances: Option<wgpu::Buffer>,
    /// How many props are placed.
    pub count: usize,
}

impl Props {
    const VERTEX: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32];
    const INSTANCE: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4];

    pub fn new(
        gpu: &Gpu,
        camera_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let device = &gpu.device;
        let air_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Props Air Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let air = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Props Air"),
            size: size_of::<AirUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let air_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Props Air Group"),
            layout: &air_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: air.as_entire_binding() }],
        });

        let module = device.create_shader_module(wgpu::include_wgsl!("../props.wgsl"));
        let (pipeline, shadow_pipeline) =
            create_pipelines(gpu, &module, [camera_layout, light_layout, &air_layout]);

        Self {
            shader: ShaderFile::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/props.wgsl")),
            air_layout,
            air,
            air_group,
            pipeline,
            shadow_pipeline,
            kinds: Vec::new(),
            instances: None,
            count: 0,
        }
    }

    /// Rebuilds the pipelines if props.wgsl was saved. See `Grass::reload_if_changed`.
    pub fn reload_if_changed(
        &mut self,
        gpu: &Gpu,
        camera_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
    ) -> Option<Result<(), String>> {
        let source = self.shader.changed_source()?;
        let built = try_build(&gpu.device, || {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("props.wgsl (reloaded)"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            create_pipelines(gpu, &module, [camera_layout, light_layout, &self.air_layout])
        });
        Some(built.map(|pipelines| (self.pipeline, self.shadow_pipeline) = pipelines))
    }

    /// Replaces every prop. Each placement is an index into `meshes` and
    /// where to put that copy.
    pub fn set(
        &mut self,
        gpu: &Gpu,
        meshes: &[MeshBuilder],
        mut placements: Vec<(usize, PropInstance)>,
    ) {
        let device = &gpu.device;
        let chunks_per_side = (terrain::SIZE / CHUNK).ceil();
        let cell_of = |instance: &PropInstance| {
            let along = |v: f32| {
                ((v + terrain::SIZE / 2.0) / CHUNK).floor().clamp(0.0, chunks_per_side - 1.0) as u32
            };
            (along(instance.position[0]), along(instance.position[2]))
        };
        // Copies of one mesh in one chunk end up side by side: one draw call.
        placements.sort_by_key(|(mesh, instance)| (*mesh, cell_of(instance)));

        self.kinds = meshes
            .iter()
            .map(|mesh| Kind {
                vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Prop Vertices"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Prop Indices"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                index_count: mesh.indices.len() as u32,
                chunks: Vec::new(),
            })
            .collect();

        for (index, (mesh, instance)) in placements.iter().enumerate() {
            let (index, cell) = (index as u32, cell_of(instance));
            let chunks = &mut self.kinds[*mesh].chunks;
            match chunks.last_mut() {
                Some(chunk) if chunk.cell == cell => chunk.instances.end = index + 1,
                _ => {
                    let middle = |c: u32| (c as f32 + 0.5) * CHUNK - terrain::SIZE / 2.0;
                    let (x, z) = (middle(cell.0), middle(cell.1));
                    chunks.push(Chunk {
                        cell,
                        centre: vec3(x, terrain::height(x, z), z),
                        instances: index..index + 1,
                    });
                }
            }
        }

        let raw: Vec<PropInstance> = placements.iter().map(|(_, instance)| *instance).collect();
        self.count = raw.len();
        self.instances = (!raw.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Prop Instances"),
                contents: bytemuck::cast_slice(&raw),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let triangles: usize = placements.iter().map(|(mesh, _)| meshes[*mesh].indices.len() / 3).sum();
        log::info!("{} props placed, {} meshes, {triangles} triangles in all", self.count, meshes.len());
    }

    /// `wind` is direction x, direction z, strength, speed.
    pub fn prepare(&self, gpu: &Gpu, wind: [f32; 4], seconds: f32) {
        let uniform = AirUniform { wind, time: [seconds, 0.0, 0.0, 0.0] };
        gpu.queue.write_buffer(&self.air, 0, bytemuck::bytes_of(&uniform));
    }

    /// Draws chunks whose bounding sphere (centre, radius) `visible` accepts.
    /// Camera (group 1) and light (group 2) must already be bound.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, visible: impl Fn(Vec3, f32) -> bool) {
        pass.set_pipeline(&self.pipeline);
        self.draw_chunks(pass, visible);
    }

    /// The same into the shadow map. `light_camera` is the light's point of view.
    pub fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        light_camera: &wgpu::BindGroup,
        visible: impl Fn(Vec3, f32) -> bool,
    ) {
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(1, light_camera, &[]);
        self.draw_chunks(pass, visible);
    }

    fn draw_chunks(&self, pass: &mut wgpu::RenderPass<'_>, visible: impl Fn(Vec3, f32) -> bool) {
        let Some(instances) = &self.instances else { return };
        pass.set_bind_group(3, &self.air_group, &[]);
        pass.set_vertex_buffer(1, instances.slice(..));
        for kind in &self.kinds {
            pass.set_vertex_buffer(0, kind.vertices.slice(..));
            pass.set_index_buffer(kind.indices.slice(..), wgpu::IndexFormat::Uint32);
            for chunk in &kind.chunks {
                if visible(chunk.centre, CHUNK_RADIUS) {
                    pass.draw_indexed(0..kind.index_count, 0, chunk.instances.clone());
                }
            }
        }
    }
}

fn create_pipelines(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    [camera, light, air]: [&wgpu::BindGroupLayout; 3],
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let device = &gpu.device;
    let buffers = [
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<PropVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Props::VERTEX,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<PropInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Props::INSTANCE,
        }),
    ];
    let render = |shadow: bool| {
        let label = if shadow { "Props Shadow Pipeline" } else { "Props Pipeline" };
        let groups = [None, Some(camera), (!shadow).then_some(light), Some(air)];
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let targets = [Some(gpu.config.format.into())];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(if shadow { "vs_shadow" } else { "vs_main" }),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            fragment: (!shadow).then_some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            // Leaves and petals have no thickness: both sides are drawn.
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: if shadow {
                    wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 }
                } else {
                    wgpu::DepthBiasState::default()
                },
            }),
            multisample: wgpu::MultisampleState {
                count: if shadow { 1 } else { SAMPLE_COUNT },
                ..Default::default()
            },
            cache: None,
            multiview_mask: None,
        })
    };
    (render(false), render(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_have_no_padding() {
        assert_eq!(size_of::<PropVertex>(), 40);
        assert_eq!(size_of::<PropInstance>(), 32);
    }

    #[test]
    fn culling_keeps_what_is_in_front_and_drops_what_is_behind() {
        let view = glam::camera::rh::view::look_at_mat4(vec3(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(0.8, 1.5, 0.1, 500.0);
        let view_proj = proj * view;
        assert!(sphere_visible(view_proj, Vec3::ZERO, 1.0));
        assert!(!sphere_visible(view_proj, vec3(0.0, 10.0, 60.0), 5.0));
        assert!(!sphere_visible(view_proj, vec3(200.0, 0.0, 0.0), 5.0));
        // Off screen, but big enough to poke into view.
        assert!(sphere_visible(view_proj, vec3(0.0, 10.0, 12.0), 5.0));
    }

    #[test]
    fn ribbon_ends_where_its_length_says() {
        let mut builder = MeshBuilder::default();
        let ribbon = Ribbon { length: 2.0, lift: 0.0, curl: 0.0, ..Default::default() };
        let (end, heading) = builder.ribbon(&ribbon);
        assert!(end.abs_diff_eq(Vec3::X * 2.0, 1e-5) && heading.abs_diff_eq(Vec3::X, 1e-5));
        assert_eq!(builder.indices.len(), ribbon.segments * 4 * 3);
        assert!(builder.indices.iter().all(|&i| (i as usize) < builder.vertices.len()));

        // Sway is nothing at the ground and `flex` at the top.
        let mut upright = MeshBuilder::default();
        upright.stalk(Vec3::ZERO, Vec3::Y, Vec3::ZERO, 0.1, Vec3::ONE);
        let upright = upright.with_sway(0.5);
        let sway = |y: f32| {
            upright.vertices.iter().find(|v| (v.position[1] - y).abs() < 1e-4).unwrap().sway
        };
        assert_eq!(sway(0.0), 0.0);
        assert!((sway(1.0) - 0.5).abs() < 1e-5);
    }
}
