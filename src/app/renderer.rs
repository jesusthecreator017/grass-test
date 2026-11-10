use crate::app::camera::{Camera, Orbit};
use crate::app::gpu::{DEPTH_FORMAT, Gpu, SAMPLE_COUNT};
use crate::app::grass::Grass;
use crate::app::instance::{InstanceRaw, Instances};
use crate::app::light::Light;
use crate::app::mesh::Mesh;
use crate::app::model::{self, Animator};
use crate::app::props::{self, Props};
use crate::app::texture::Texture;
use crate::app::vertex::Vertex;
use std::sync::Arc;

/// A shader source file watched for edits, so debug builds can reload it
/// while running.
pub struct ShaderFile {
    path: &'static str,
    seen: Option<std::time::SystemTime>,
}

impl ShaderFile {
    /// Assumes the shader compiled into the program is what is on disk now.
    pub fn new(path: &'static str) -> Self {
        let mut file = Self { path, seen: None };
        file.seen = file.modified();
        file
    }

    fn modified(&self) -> Option<std::time::SystemTime> {
        std::fs::metadata(self.path).and_then(|meta| meta.modified()).ok()
    }

    /// The file's contents, once, each time it is saved. Debug builds only.
    pub fn changed_source(&mut self) -> Option<String> {
        if !cfg!(debug_assertions) {
            return None;
        }
        let modified = self.modified()?;
        if Some(modified) == self.seen {
            return None;
        }
        self.seen = Some(modified);
        std::fs::read_to_string(self.path).ok()
    }
}

/// Runs GPU object creation that might be invalid (a shader with a typo)
/// and returns the error instead of letting wgpu panic.
pub fn try_build<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> Result<T, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = build();
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(built),
    }
}

/// Component: what the editor calls an entity.
pub struct Name(pub String);

/// Component: where an entity is.
#[derive(Clone, Copy)]
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl Transform {
    pub fn at(position: glam::Vec3) -> Self {
        Self { position, rotation: glam::Quat::IDENTITY, scale: glam::Vec3::ONE }
    }

    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

/// Component: what an entity looks like. Anything with a `Transform` and a
/// `Renderable` gets drawn; add an `Animator` and it is skinned too.
pub struct Renderable {
    pub mesh: Arc<Mesh>,
    pub texture: Arc<Texture>,
    pub tint: glam::Vec4,
}

/// A run of instances that share mesh, texture and joints: one draw call.
struct Batch {
    mesh: Arc<Mesh>,
    texture: Arc<Texture>,
    joints: Option<wgpu::BindGroup>,
    instances: std::ops::Range<u32>,
}

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    /// The light's point of view, bound in the camera slot for the shadow pass.
    shadow_camera: Camera,
    pub light: Light,
    pub sky: [f32; 3],
    /// Drawn brighter and warmer, so the editor's selection is visible in the world.
    // ponytail: a tint. Draw a proper outline pass if this gets hard to see.
    pub selected: Option<hecs::Entity>,
    /// Why the last shader reload was rejected, if it was.
    pub shader_error: Option<String>,
    shader: ShaderFile,
    pub grass: Grass,
    pub props: Props,
    instances: Instances,
    pub texture_layout: wgpu::BindGroupLayout,
    camera_layout: wgpu::BindGroupLayout,
    light_layout: wgpu::BindGroupLayout,
    pub joints_layout: wgpu::BindGroupLayout,
    /// Bound for everything that has no skeleton.
    no_joints: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let texture_layout = Texture::bind_group_layout(device);
        let camera_layout = Camera::bind_group_layout(device);
        let light_layout = Light::bind_group_layout(device);
        let joints_layout = Animator::bind_group_layout(device);

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let layouts = [&texture_layout, &camera_layout, &light_layout, &joints_layout];
        let (pipeline, shadow_pipeline) = create_pipelines(gpu, &shader, layouts);

        Self {
            pipeline,
            shadow_pipeline,
            camera: Camera::new(device, &camera_layout),
            shadow_camera: Camera::new(device, &camera_layout),
            light: Light::new(
                device,
                &light_layout,
                glam::vec3(-0.6, -1.0, -0.4),
                glam::vec3(1.0, 0.96, 0.9),
            ),
            sky: [0.25, 0.5, 0.85],
            selected: None,
            shader_error: None,
            shader: ShaderFile::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shader.wgsl")),
            grass: Grass::new(gpu, &camera_layout, &light_layout),
            props: Props::new(gpu, &camera_layout, &light_layout),
            instances: Instances::new(device, 256),
            no_joints: model::joint_buffer(device, &joints_layout).1,
            texture_layout,
            camera_layout,
            light_layout,
            joints_layout,
        }
    }

    /// Debug builds rebuild pipelines when a shader file is saved; a bad shader
    /// is reported and the last good one stays.
    pub fn reload_shaders_if_changed(&mut self, gpu: &Gpu) {
        let mut outcomes = Vec::new();

        if let Some(source) = self.shader.changed_source() {
            let layouts =
                [&self.texture_layout, &self.camera_layout, &self.light_layout, &self.joints_layout];
            let built = try_build(&gpu.device, || {
                let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("shader.wgsl (reloaded)"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                create_pipelines(gpu, &shader, layouts)
            });
            outcomes.push(built.map(|pipelines| (self.pipeline, self.shadow_pipeline) = pipelines));
        }
        outcomes.extend(self.grass.reload_if_changed(gpu, &self.camera_layout, &self.light_layout));
        outcomes.extend(self.props.reload_if_changed(gpu, &self.camera_layout, &self.light_layout));

        for outcome in outcomes {
            match outcome {
                Ok(()) => {
                    log::info!("shader reloaded");
                    self.shader_error = None;
                }
                Err(error) => {
                    log::error!("shader reload failed: {error}");
                    self.shader_error = Some(error);
                }
            }
        }
    }

    /// Draws every renderable entity: first into the shadow map, then into
    /// `target` (which the multisampled colour buffer resolves to).
    pub fn render(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        world: &hecs::World,
        orbit: &Orbit,
    ) {
        let batches = self.prepare(gpu, world);

        let view_proj = orbit.view_proj(gpu.aspect());
        self.grass.prepare(gpu, world, view_proj, orbit.eye());
        self.grass.generate(encoder);
        let (wind, seconds) = self.grass.wind();
        self.props.prepare(gpu, wind, seconds);

        self.camera.upload(&gpu.queue, view_proj, orbit.eye());
        self.shadow_camera.upload(&gpu.queue, self.light.view_proj(orbit.target), orbit.eye());
        self.light.upload(&gpu.queue, orbit.target);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.light.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                color_attachments: &[],
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(1, &self.shadow_camera.bind_group, &[]);
            self.draw(&mut pass, &batches, false);
            self.grass.draw_shadow(&mut pass, &self.shadow_camera.bind_group);
            // Only props near enough to the view to land in the shadow map.
            self.props.draw_shadow(&mut pass, &self.shadow_camera.bind_group, |centre, radius| {
                centre.distance(orbit.target) < 45.0 + radius
            });
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.sky[0] as f64,
                            g: self.sky[1] as f64,
                            b: self.sky[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(1, &self.camera.bind_group, &[]);
            pass.set_bind_group(2, &self.light.bind_group, &[]);
            self.draw(&mut pass, &batches, true);
            self.grass.draw_ground(&mut pass);
            self.props.draw(&mut pass, |centre, radius| {
                props::sphere_visible(view_proj, centre, radius)
            });
            self.grass.draw(&mut pass);
        }
    }

    /// Uploads this frame's instances and joint poses, and groups the
    /// instances into as few draw calls as possible.
    fn prepare(&mut self, gpu: &Gpu, world: &hecs::World) -> Vec<Batch> {
        let mut query =
            world.query::<(hecs::Entity, &Transform, &Renderable, Option<&Animator>)>();
        let mut items: Vec<_> = query
            .iter()
            .map(|(entity, transform, renderable, animator)| {
                if let Some(animator) = animator {
                    animator.upload(&gpu.queue);
                }
                // Same mesh and texture sort together; a skinned entity can't
                // share a draw, so its id keeps it apart.
                let key = (
                    Arc::as_ptr(&renderable.mesh) as usize,
                    Arc::as_ptr(&renderable.texture) as usize,
                    animator.map_or(0, |_| entity.to_bits().get()),
                );
                let mut tint = renderable.tint;
                if self.selected == Some(entity) {
                    tint *= glam::vec4(2.6, 1.8, 0.8, 1.0);
                }
                let raw = InstanceRaw {
                    model: transform.matrix().to_cols_array_2d(),
                    tint: tint.to_array(),
                };
                (key, raw, renderable, animator)
            })
            .collect();
        items.sort_unstable_by_key(|item| item.0);

        let raw: Vec<InstanceRaw> = items.iter().map(|item| item.1).collect();
        self.instances.upload(&gpu.device, &gpu.queue, &raw);

        let mut batches: Vec<Batch> = Vec::new();
        let mut previous = None;
        for (index, (key, _, renderable, animator)) in items.into_iter().enumerate() {
            let index = index as u32;
            match batches.last_mut() {
                Some(batch) if previous == Some(key) => batch.instances.end = index + 1,
                _ => batches.push(Batch {
                    mesh: renderable.mesh.clone(),
                    texture: renderable.texture.clone(),
                    joints: animator.map(|a| a.bind_group.clone()),
                    instances: index..index + 1,
                }),
            }
            previous = Some(key);
        }
        batches
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, batches: &[Batch], textured: bool) {
        pass.set_vertex_buffer(1, self.instances.slice());
        for batch in batches {
            if textured {
                pass.set_bind_group(0, &batch.texture.bind_group, &[]);
            }
            pass.set_bind_group(3, batch.joints.as_ref().unwrap_or(&self.no_joints), &[]);
            batch.mesh.draw(pass, batch.instances.clone());
        }
    }
}

/// The main and shadow pipelines. `layouts` is texture, camera, light, joints.
fn create_pipelines(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    [texture, camera, light, joints]: [&wgpu::BindGroupLayout; 4],
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let all = [Some(texture), Some(camera), Some(light), Some(joints)];
    // Shadows need neither the texture nor the light they are made for.
    let shadow = [None, Some(camera), None, Some(joints)];
    (
        create_pipeline(gpu, shader, "Main Pipeline", &all, false),
        create_pipeline(gpu, shader, "Shadow Pipeline", &shadow, true),
    )
}

fn create_pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    label: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
    shadow: bool,
) -> wgpu::RenderPipeline {
    let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        immediate_size: 0,
    });
    let targets = [Some(gpu.config.format.into())];

    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(if shadow { "vs_shadow" } else { "vs_main" }),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(Instances::layout())],
        },
        // The shadow pass only writes depth.
        fragment: (!shadow).then_some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            // Pushes shadow casters away from the light a little, so surfaces
            // don't shadow themselves ("shadow acne").
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
}
