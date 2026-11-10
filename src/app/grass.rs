//! Procedural grass: the CPU side of grass.wgsl. Owns the buffers the compute
//! shader fills with blades and the pipelines that fill and draw them.

use crate::app::gpu::{DEPTH_FORMAT, Gpu, SAMPLE_COUNT};
use crate::app::renderer::{ShaderFile, Transform, try_build};
use crate::app::terrain;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use wgpu::util::DeviceExt;

/// Max blades per level of detail per frame; extras are silently skipped.
// ponytail: fixed buffers; raise these if grass goes missing in patches.
pub const NEAR_CAPACITY: u64 = 700_000;
pub const FAR_CAPACITY: u64 = 700_000;
const BLADE_SIZE: u64 = 64;
const MAX_PUSHERS: usize = 8;
/// The painted density map covers the field with this many texels a side.
const DENSITY_SIZE: u32 = 512;

/// Component: this entity flattens the grass within `0` metres of it.
// ponytail: only the first 8 count; sort by camera distance if that bites.
pub struct GrassPusher(pub f32);

// States of the blade-count readback.
const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;

pub const TYPE_NAMES: [&str; 4] = ["Short", "Regular", "Tall", "Grain"];

/// One kind of grass. The shape numbers multiply the settings all kinds
/// share, so "regular" is all ones.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrassType {
    pub height: f32,
    pub width: f32,
    pub tilt: f32,
    pub bend: f32,
    /// Share of this kind's blades that grow.
    pub density: f32,
    /// How much more (or less) than regular grass the wind moves it.
    pub wind: f32,
    pub color: [f32; 3],
    /// How far blades take on `color` instead of the shared root and tip colours.
    pub color_amount: f32,
    /// Width in metres of the ear at the top. Grain only.
    pub ear: f32,
    /// How much of the field this kind covers. Regular gets what is left.
    pub coverage: f32,
}

impl Default for GrassType {
    fn default() -> Self {
        Self {
            height: 1.0,
            width: 1.0,
            tilt: 1.0,
            bend: 1.0,
            density: 1.0,
            wind: 1.0,
            color: [0.3, 0.52, 0.12],
            color_amount: 0.0,
            ear: 0.0,
            coverage: 0.0,
        }
    }
}

/// Everything about the grass that can be tuned while the game runs.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrassSettings {
    pub enabled: bool,
    pub cast_shadows: bool,
    /// Share of the blades that grow: 1 is a thick lawn, lower is sparser.
    pub density: f32,
    /// Metres between blades. Halving it quadruples the blade count.
    pub spacing: f32,
    /// Blades closer than this get full detail and full density.
    pub near_radius: f32,
    /// Beyond this there is no grass, only the ground's colour.
    pub far_radius: f32,
    pub height: f32,
    pub width: f32,
    pub tilt: f32,
    pub bend: f32,
    pub clump_size: f32,
    pub clump_height: f32,
    pub clump_facing: f32,
    pub clump_color: f32,
    pub root_color: [f32; 3],
    pub tip_color: [f32; 3],
    pub occlusion: f32,
    pub translucency: f32,
    pub specular: f32,
    /// Degrees, 0 = along +X.
    pub wind_angle: f32,
    pub wind_strength: f32,
    pub wind_speed: f32,
    /// Metres across of the patches the field varies in.
    pub patch_size: f32,
    pub patch_height: f32,
    /// How sparse the thin patches get.
    pub patch_thinning: f32,
    pub dry_color: [f32; 3],
    pub dry_amount: f32,
    pub flower_color: [f32; 3],
    pub flower2_color: [f32; 3],
    pub flower_amount: f32,
    pub flower_size: f32,
    /// Metres across of the dirt paths; 0 for none.
    pub path_width: f32,
    /// How much of the field is worn through to dirt.
    pub worn: f32,
    pub soil_color: [f32; 3],
    /// Short, regular, tall, grain: see `TYPE_NAMES`.
    pub types: [GrassType; 4],
    /// How far the odd blade stands above its neighbours.
    pub stragglers: f32,
    /// How many clumps grow as tall tussocks.
    pub tussocks: f32,
    /// How much the height swells and dips every few metres.
    pub swell: f32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cast_shadows: true,
            density: 0.55,
            spacing: 0.07,
            near_radius: 32.0,
            far_radius: 95.0,
            height: 0.75,
            width: 0.055,
            tilt: 0.55,
            bend: 0.18,
            clump_size: 1.4,
            clump_height: 0.35,
            clump_facing: 0.6,
            clump_color: 0.18,
            root_color: [0.03, 0.11, 0.02],
            tip_color: [0.3, 0.52, 0.12],
            occlusion: 0.45,
            translucency: 0.6,
            specular: 0.25,
            wind_angle: 30.0,
            wind_strength: 0.6,
            wind_speed: 1.0,
            patch_size: 14.0,
            patch_height: 0.35,
            patch_thinning: 0.3,
            dry_color: [0.5, 0.42, 0.17],
            dry_amount: 0.35,
            flower_color: [0.95, 0.95, 0.88],
            flower2_color: [0.95, 0.7, 0.12],
            flower_amount: 0.0,
            flower_size: 0.05,
            path_width: 2.4,
            worn: 0.3,
            soil_color: [0.3, 0.19, 0.1],
            types: [
                GrassType {
                    height: 0.3,
                    width: 0.8,
                    tilt: 0.6,
                    bend: 0.6,
                    wind: 0.5,
                    color: [0.25, 0.5, 0.12],
                    color_amount: 0.35,
                    coverage: 0.25,
                    ..Default::default()
                },
                GrassType::default(),
                GrassType {
                    height: 1.6,
                    width: 1.5,
                    tilt: 1.25,
                    bend: 1.6,
                    density: 0.6,
                    wind: 1.2,
                    color: [0.08, 0.3, 0.12],
                    color_amount: 0.4,
                    coverage: 0.3,
                    ..Default::default()
                },
                GrassType {
                    height: 2.0,
                    width: 0.7,
                    tilt: 0.45,
                    bend: 0.6,
                    density: 0.5,
                    wind: 1.5,
                    color: [0.75, 0.6, 0.25],
                    color_amount: 0.9,
                    ear: 0.06,
                    coverage: 0.3,
                },
            ],
            stragglers: 0.5,
            tussocks: 0.4,
            swell: 0.3,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    wind: [f32; 4],
    lod: [f32; 4],
    shape: [f32; 4],
    clump: [f32; 4],
    root: [f32; 4],
    tip: [f32; 4],
    misc: [f32; 4],
    dry: [f32; 4],
    flower: [f32; 4],
    flower2: [f32; 4],
    vary: [f32; 4],
    ground: [f32; 4],
    soil: [f32; 4],
    vary2: [f32; 4],
    types: [[f32; 4]; 12],
    pushers: [[f32; 4]; MAX_PUSHERS],
}

pub struct Grass {
    pub settings: GrassSettings,
    shader: ShaderFile,
    compute_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    uniform: wgpu::Buffer,
    /// Indirect draw arguments for the near and far draws. The compute
    /// shader counts blades straight into them.
    draws: wgpu::Buffer,
    compute_group: wgpu::BindGroup,
    near_group: wgpu::BindGroup,
    far_group: wgpu::BindGroup,
    /// Painted in the editor and saved.
    density: Vec<u8>,
    /// Where scenery stands. The grass sees the lesser of the two.
    blocked: Vec<u8>,
    density_texture: wgpu::Texture,
    density_dirty: bool,
    start: std::time::Instant,
    /// Workgroups per side for this frame's dispatch.
    groups: u32,
    /// Blades the compute shader wanted to draw (near, far), a frame or two
    /// ago. Read back from the GPU only so the editor can show it.
    pub blade_counts: [u32; 2],
    readback: wgpu::Buffer,
    readback_state: Arc<AtomicU8>,
    copy_queued: bool,
}

impl Grass {
    pub fn new(
        gpu: &Gpu,
        camera_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let device = &gpu.device;
        let buffer_entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let uniform_type = wgpu::BufferBindingType::Uniform;
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };

        let compute = wgpu::ShaderStages::COMPUTE;
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grass Compute Layout"),
            entries: &[
                buffer_entry(0, compute, uniform_type),
                buffer_entry(1, compute, storage(false)),
                buffer_entry(2, compute, storage(false)),
                buffer_entry(3, compute, storage(false)),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let draw_stages = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grass Draw Layout"),
            entries: &[
                buffer_entry(0, draw_stages, uniform_type),
                buffer_entry(1, wgpu::ShaderStages::VERTEX, storage(true)),
                // The ground reads the density map to show dirt where grass is cleared.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grass Uniform"),
            size: size_of::<GrassUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blades = |label, capacity: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity * BLADE_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let near_blades = blades("Near Blades", NEAR_CAPACITY);
        let far_blades = blades("Far Blades", FAR_CAPACITY);
        let draws = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grass Draw Arguments"),
            contents: bytemuck::cast_slice(&EMPTY_DRAWS),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grass Count Readback"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let density = vec![255u8; (DENSITY_SIZE * DENSITY_SIZE) as usize];
        let density_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Grass Density"),
            size: wgpu::Extent3d {
                width: DENSITY_SIZE,
                height: DENSITY_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let density_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Grass Density Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        let compute_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grass Compute Group"),
            layout: &compute_layout,
            entries: &[
                entry(0, &uniform),
                entry(1, &near_blades),
                entry(2, &far_blades),
                entry(3, &draws),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &density_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&density_sampler),
                },
            ],
        });
        let density_view = density_texture.create_view(&Default::default());
        let draw_group = |label, blades: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &draw_layout,
                entries: &[
                    entry(0, &uniform),
                    entry(1, blades),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&density_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&density_sampler),
                    },
                ],
            })
        };
        let near_group = draw_group("Near Grass Group", &near_blades);
        let far_group = draw_group("Far Grass Group", &far_blades);

        let module = device.create_shader_module(wgpu::include_wgsl!("../grass.wgsl"));
        let pipelines = create_pipelines(
            gpu,
            &module,
            &compute_layout,
            [camera_layout, light_layout, &draw_layout],
        );

        Self {
            settings: GrassSettings::default(),
            shader: ShaderFile::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/grass.wgsl")),
            compute_layout,
            draw_layout,
            pipelines,
            uniform,
            draws,
            compute_group,
            near_group,
            far_group,
            blocked: density.clone(),
            density,
            density_texture,
            density_dirty: true,
            start: std::time::Instant::now(),
            groups: 0,
            blade_counts: [0; 2],
            readback,
            readback_state: Arc::new(AtomicU8::new(IDLE)),
            copy_queued: false,
        }
    }

    /// Rebuilds the pipelines if grass.wgsl was saved. On a shader that
    /// doesn't compile, returns the error and keeps the old pipelines.
    pub fn reload_if_changed(
        &mut self,
        gpu: &Gpu,
        camera_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
    ) -> Option<Result<(), String>> {
        let source = self.shader.changed_source()?;
        let built = try_build(&gpu.device, || {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grass.wgsl (reloaded)"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            create_pipelines(
                gpu,
                &module,
                &self.compute_layout,
                [camera_layout, light_layout, &self.draw_layout],
            )
        });
        Some(built.map(|pipelines| self.pipelines = pipelines))
    }

    /// Raises (positive `amount`) or lowers the painted grass density in a
    /// soft-edged circle on the ground.
    pub fn paint(&mut self, centre: Vec3, radius: f32, amount: f32) {
        stamp(&mut self.density, centre, radius, |texel, falloff| {
            (texel as f32 + amount * falloff * 255.0).clamp(0.0, 255.0) as u8
        });
        self.density_dirty = true;
    }

    /// Stops grass growing in a circle, whatever is painted there. For
    /// scenery that stands on the ground; unlike paint, it isn't saved.
    pub fn block(&mut self, centre: Vec3, radius: f32) {
        // Bare in the middle, grass returning over the outer part.
        stamp(&mut self.blocked, centre, radius, |texel, falloff| {
            texel.min((255.0 * (1.0 - (falloff * 2.5).min(1.0))) as u8)
        });
        self.density_dirty = true;
    }

    pub fn save_density(&self, path: &str) -> Result<(), String> {
        image::save_buffer(path, &self.density, DENSITY_SIZE, DENSITY_SIZE, image::ColorType::L8)
            .map_err(|err| err.to_string())
    }

    /// Replaces the painted density with a saved one. No file is not an
    /// error: nothing has been saved yet.
    pub fn load_density(&mut self, path: &str) {
        if !std::path::Path::new(path).exists() {
            return;
        }
        match image::open(path).map(|image| image.to_luma8()) {
            Ok(image) if image.dimensions() == (DENSITY_SIZE, DENSITY_SIZE) => {
                self.density = image.into_raw();
                self.density_dirty = true;
            }
            Ok(_) => log::warn!("{path} is not {DENSITY_SIZE}x{DENSITY_SIZE}, ignored"),
            Err(err) => log::warn!("can't load {path}: {err}"),
        }
    }

    /// Uploads this frame's settings and empties the blade counters.
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        world: &hecs::World,
        view_proj: glam::Mat4,
        eye: Vec3,
    ) {
        if std::mem::take(&mut self.density_dirty) {
            let combined: Vec<u8> =
                self.density.iter().zip(&self.blocked).map(|(a, b)| *a.min(b)).collect();
            gpu.queue.write_texture(
                self.density_texture.as_image_copy(),
                &combined,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(DENSITY_SIZE),
                    rows_per_image: Some(DENSITY_SIZE),
                },
                self.density_texture.size(),
            );
        }

        // Collect the blade counts asked for on an earlier frame.
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if self.readback_state.load(Ordering::Acquire) == READY {
            if let Ok(data) = self.readback.get_mapped_range(..) {
                let count = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                self.blade_counts = [count(4), count(20)];
            }
            self.readback.unmap();
            self.readback_state.store(IDLE, Ordering::Release);
        }
        if !self.settings.enabled {
            self.blade_counts = [0; 2];
        }

        let s = &self.settings;
        // Guard the sliders: a zero spacing would ask for infinite blades.
        let spacing = s.spacing.max(0.03);
        let far_radius = s.far_radius.max(s.near_radius);

        let mut pushers = [[0.0; 4]; MAX_PUSHERS];
        let mut pusher_count = 0;
        for (transform, pusher) in
            world.query::<(&Transform, &GrassPusher)>().iter().take(MAX_PUSHERS)
        {
            pushers[pusher_count] = transform.position.extend(pusher.0).to_array();
            pusher_count += 1;
        }

        let wind = Vec2::from_angle(s.wind_angle.to_radians());
        let uniform = GrassUniform {
            view_proj: view_proj.to_cols_array_2d(),
            eye: eye.extend(self.start.elapsed().as_secs_f32()).to_array(),
            wind: [wind.x, wind.y, s.wind_strength, s.wind_speed],
            lod: [s.near_radius, far_radius, spacing, terrain::SIZE],
            shape: [s.height, s.width, s.tilt, s.bend],
            clump: [s.clump_size.max(0.1), s.clump_height, s.clump_facing, s.clump_color],
            root: [s.root_color[0], s.root_color[1], s.root_color[2], s.occlusion],
            tip: [s.tip_color[0], s.tip_color[1], s.tip_color[2], s.translucency],
            misc: [s.specular, pusher_count as f32, 0.0, 0.0],
            dry: [s.dry_color[0], s.dry_color[1], s.dry_color[2], s.dry_amount],
            flower: [s.flower_color[0], s.flower_color[1], s.flower_color[2], s.flower_amount],
            flower2: [s.flower2_color[0], s.flower2_color[1], s.flower2_color[2], s.flower_size],
            vary: [s.patch_height, s.patch_thinning, s.patch_size.max(1.0), s.density],
            ground: [s.path_width, s.worn, 0.0, 0.0],
            soil: [s.soil_color[0], s.soil_color[1], s.soil_color[2], 0.0],
            vary2: [s.stragglers, s.tussocks, s.swell, 0.0],
            types: std::array::from_fn(|i| {
                let t = &s.types[i / 3];
                match i % 3 {
                    0 => [t.height, t.width, t.tilt, t.bend],
                    1 => [t.density, t.wind, t.color_amount, t.ear],
                    _ => [t.color[0], t.color[1], t.color[2], t.coverage],
                }
            }),
            pushers,
        };
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        gpu.queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&EMPTY_DRAWS));

        // Enough 8x8 workgroups to cover the larger of the two grids: the
        // near one at full density, the far one at every second cell.
        let cells = (2.0 * s.near_radius / spacing).max(far_radius / spacing).ceil() as u32;
        self.groups = cells.div_ceil(8);
    }

    /// Runs the compute shader that places this frame's blades.
    pub fn generate(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.settings.enabled {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Grass Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compute);
            pass.set_bind_group(0, &self.compute_group, &[]);
            pass.dispatch_workgroups(self.groups, self.groups, 2);
        }
        // One readback at a time: the buffer can't be copied into while the
        // last copy is still being read.
        if self.readback_state.load(Ordering::Acquire) == IDLE {
            encoder.copy_buffer_to_buffer(&self.draws, 0, &self.readback, 0, 32u64);
            self.copy_queued = true;
        }
    }

    /// Call once the frame's commands are submitted.
    pub fn after_submit(&mut self) {
        if !std::mem::take(&mut self.copy_queued) {
            return;
        }
        self.readback_state.store(PENDING, Ordering::Release);
        let state = self.readback_state.clone();
        self.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    /// Draws the near blades into the shadow map.
    // ponytail: only camera-visible blades cast shadows, so shadows can pop at
    // screen edges; generate a light-culled set if it shows.
    pub fn draw_shadow(&self, pass: &mut wgpu::RenderPass<'_>, light_camera: &wgpu::BindGroup) {
        if !self.settings.enabled || !self.settings.cast_shadows {
            return;
        }
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(1, light_camera, &[]);
        pass.set_bind_group(3, &self.near_group, &[]);
        pass.draw_indirect(&self.draws, 0);
    }

    /// Wind direction (x, z), strength and speed, and the time in seconds:
    /// what else needs to move with the grass.
    pub fn wind(&self) -> ([f32; 4], f32) {
        let s = &self.settings;
        let direction = Vec2::from_angle(s.wind_angle.to_radians());
        ([direction.x, direction.y, s.wind_strength, s.wind_speed], self.start.elapsed().as_secs_f32())
    }

    /// Draws the ground (turf, dirt, paths). Same bindings as `draw`.
    // ponytail: ground is not in the shadow map, so hills cast no shadows.
    pub fn draw_ground(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipelines.ground);
        pass.set_bind_group(3, &self.near_group, &[]);
        // Two triangles for each of 200 x 200 squares; see vs_ground.
        pass.draw(0..6 * 200 * 200, 0..1);
    }

    /// Draws the blades. The caller has the camera in group 1 and the light
    /// in group 2 already.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.settings.enabled {
            return;
        }
        pass.set_pipeline(&self.pipelines.draw);
        pass.set_bind_group(3, &self.near_group, &[]);
        pass.draw_indirect(&self.draws, 0);
        pass.set_bind_group(3, &self.far_group, &[]);
        pass.draw_indirect(&self.draws, 16);
    }
}

/// Vertex count, instance count, first vertex, first instance - for the near
/// draw (15 vertices a blade) and the far one (7).
const EMPTY_DRAWS: [[u32; 4]; 2] = [[15, 0, 0, 0], [7, 0, 0, 0]];

/// Applies `apply(texel, falloff)` to every texel of a density map within
/// `radius` metres of `centre`; falloff runs from 1 in the middle to 0 at the rim.
fn stamp(map: &mut [u8], centre: Vec3, radius: f32, apply: impl Fn(u8, f32) -> u8) {
    let to_texel = DENSITY_SIZE as f32 / terrain::SIZE;
    let centre = (Vec2::new(centre.x, centre.z) + terrain::SIZE / 2.0) * to_texel;
    let radius = (radius * to_texel).max(0.5);
    let reach = radius.ceil() as i32;
    let size = DENSITY_SIZE as i32;
    for y in -reach..=reach {
        for x in -reach..=reach {
            let (tx, ty) = (centre.x as i32 + x, centre.y as i32 + y);
            if !(0..size).contains(&tx) || !(0..size).contains(&ty) {
                continue;
            }
            let texel_centre = Vec2::new(tx as f32 + 0.5, ty as f32 + 0.5);
            let falloff = 1.0 - (texel_centre.distance(centre) / radius).min(1.0);
            let texel = &mut map[(ty * size + tx) as usize];
            *texel = apply(*texel, falloff);
        }
    }
}

struct Pipelines {
    compute: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    ground: wgpu::RenderPipeline,
}

fn create_pipelines(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    compute_layout: &wgpu::BindGroupLayout,
    [camera, light, draw]: [&wgpu::BindGroupLayout; 3],
) -> Pipelines {
    let device = &gpu.device;
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Grass Compute Pipeline"),
        bind_group_layouts: &[Some(compute_layout)],
        immediate_size: 0,
    });
    let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Grass Compute Pipeline"),
        layout: Some(&layout),
        module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Group 0 belongs to the compute stage; drawing doesn't use it. The
    // shadow variant writes depth only, from the light, so has no light group.
    let render = |label: &str, vertex: &str, fragment: Option<&str>, blades: bool| {
        let shadow = fragment.is_none();
        let groups = [None, Some(camera), (!shadow).then_some(light), Some(draw)];
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
                entry_point: Some(vertex),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: fragment.map(|entry_point| wgpu::FragmentState {
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            primitive: if blades {
                wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    // Blades are paper thin and seen from both sides.
                    cull_mode: None,
                    ..Default::default()
                }
            } else {
                wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() }
            },
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
    Pipelines {
        compute,
        draw: render("Grass Draw Pipeline", "vs_main", Some("fs_main"), true),
        shadow: render("Grass Shadow Pipeline", "vs_main", None, true),
        ground: render("Ground Pipeline", "vs_ground", Some("fs_ground"), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_matches_the_shader_struct() {
        // mat4 + 15 vec4 + 4 types of 3 vec4 + 8 pushers, no padding anywhere.
        assert_eq!(size_of::<GrassUniform>(), 64 + 27 * 16 + MAX_PUSHERS * 16);
    }

    // The shader has its own copy of the terrain formula. If the two drift
    // apart the grass floats or sinks, so compare the text of both.
    #[test]
    fn shader_terrain_matches_rust_terrain() {
        let numbers = |source: &str, start: &str| -> Vec<String> {
            let body = &source[source.find(start).expect("function exists")..];
            let body = &body[..body.find("\n}").unwrap()];
            body.split(|c: char| !(c.is_ascii_digit() || c == '.'))
                .filter(|token| token.contains('.') && token.len() > 1)
                .map(str::to_string)
                .collect()
        };
        for (rust, wgsl) in [("pub fn height", "fn terrain_height"), ("pub fn path_distance", "fn path_distance")] {
            let rust = numbers(include_str!("terrain.rs"), rust);
            let wgsl = numbers(include_str!("../grass.wgsl"), wgsl);
            assert!(!rust.is_empty());
            assert_eq!(rust, wgsl);
        }
    }
}
