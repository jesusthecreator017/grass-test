//! The sandbox scene: a rolling field with a cube in the middle. Everything
//! game-specific lives here; `app/` is the engine it runs on.

use crate::app::audio::Audio;
use crate::app::camera::Orbit;
use crate::app::gpu::Gpu;
use crate::app::grass::GrassPusher;
use crate::app::input::Input;
use crate::app::mesh::Mesh;
use crate::app::model::{Animator, Model};
use crate::app::nav::{Cell, Grid};
use crate::app::renderer::{Name, Renderable, Renderer, Transform};
use crate::app::texture::Texture;
use crate::app::{terrain, vertex};
use crate::error::Result;
use glam::{Quat, Vec2, Vec3, Vec4, vec2, vec3};
use std::sync::Arc;
use winit::keyboard::KeyCode;

const PLAYER_SPEED: f32 = 7.0;
const WALKER_SPEED: f32 = 5.0;
const PAN_SPEED: f32 = 25.0;
/// Radius of the bare patch a tower stands in.
const TOWER_CLEARING: f32 = 1.3;

/// Component: a cube placed by clicking. Blocks its grid cell.
struct Tower;

/// Component: follows a path of cells towards `goal`.
struct Walker {
    goal: Cell,
    path: Vec<Cell>,
    next: usize,
}

/// The engine pieces game code may touch during an update.
pub struct Context<'a> {
    pub gpu: &'a Gpu,
    pub renderer: &'a mut Renderer,
    pub input: &'a Input,
    pub orbit: &'a mut Orbit,
    pub audio: &'a Audio,
}

pub struct Game {
    pub world: hecs::World,
    grid: Grid,
    /// Rocks: centre (x, z) and radius.
    obstacles: Vec<(Vec2, f32)>,
    player: hecs::Entity,
    cube: Arc<Mesh>,
    white: Arc<Texture>,
    fox: Option<Model>,
    follow: bool,
    // Set by UI buttons, acted on in the next update.
    spawn_fox: bool,
    clear_towers: bool,
}

impl Game {
    pub fn new(gpu: &Gpu, renderer: &mut Renderer) -> Result<Self> {
        let (device, queue, layout) = (&gpu.device, &gpu.queue, &renderer.texture_layout);
        let mut world = hecs::World::new();

        let white = Arc::new(Texture::solid(device, queue, layout, [255; 4]));
        let crate_texture = Arc::new(Texture::from_image(
            device,
            queue,
            layout,
            concat!(env!("CARGO_MANIFEST_DIR"), "/assets/texture.jpeg"),
        )?);
        let cube = Arc::new(Mesh::new(device, "Cube", &vertex::CUBE, &vertex::CUBE_INDICES));

        let player = world.spawn((
            Name("Cube".into()),
            GrassPusher(1.6),
            Transform::at(on_ground(0.5, 0.5, 0.5)),
            Renderable { mesh: cube.clone(), texture: crate_texture, tint: Vec4::ONE },
        ));

        // The fox only exists to exercise skinning and pathfinding; the scene
        // works without it.
        let fox = Model::load(device, queue, layout, concat!(env!("CARGO_MANIFEST_DIR"), "/assets/Fox.glb"))
            .inspect_err(|err| log::warn!("fox unavailable: {err}"))
            .ok();

        // Plants and rocks. Rocks are in the way of everything; a big plant
        // only shades out the grass under it.
        let scene = crate::flora::populate();
        for &(centre, radius) in &scene.obstacles {
            renderer.grass.block(vec3(centre.x, 0.0, centre.y), radius + 0.3);
        }
        for &(centre, radius) in &scene.clearings {
            renderer.grass.block(vec3(centre.x, 0.0, centre.y), radius);
        }
        renderer.props.set(gpu, &scene.meshes, scene.placements);

        let mut game = Self {
            world,
            grid: Grid::new((terrain::SIZE / 2.0) as i32),
            obstacles: scene.obstacles,
            player,
            cube,
            white,
            fox,
            follow: true,
            spawn_fox: false,
            clear_towers: false,
        };
        game.block_obstacles();
        Ok(game)
    }

    /// Marks the cells rocks stand on as impassable.
    fn block_obstacles(&mut self) {
        for &(centre, radius) in &self.obstacles {
            let reach = radius.ceil() as i32 + 1;
            let (cx, cz) = (centre.x.floor() as i32, centre.y.floor() as i32);
            for z in cz - reach..=cz + reach {
                for x in cx - reach..=cx + reach {
                    let (mx, mz) = Grid::center((x, z));
                    if vec2(mx, mz).distance(centre) < radius + 0.3 {
                        self.grid.block((x, z));
                    }
                }
            }
        }
    }

    /// Camera keys (Q/E turn, arrows pan, F follow). Split from `update` so the
    /// editor can move the camera while paused. Returns ground-plane (up, right).
    pub fn camera(&mut self, ctx: &mut Context, dt: f32) -> (Vec3, Vec3) {
        let input = ctx.input;
        ctx.orbit.yaw += input.axis(KeyCode::KeyE, KeyCode::KeyQ) * 1.5 * dt;
        let (forward, right) = ctx.orbit.ground_axes();
        let pan = forward * input.axis(KeyCode::ArrowDown, KeyCode::ArrowUp)
            + right * input.axis(KeyCode::ArrowLeft, KeyCode::ArrowRight);
        if pan != Vec3::ZERO {
            self.follow = false;
            let target = ctx.orbit.target + pan.normalize() * PAN_SPEED * dt;
            let limit = terrain::SIZE / 2.0;
            let (x, z) = (target.x.clamp(-limit, limit), target.z.clamp(-limit, limit));
            ctx.orbit.target = on_ground(x, z, 0.0);
        }
        if input.pressed(KeyCode::KeyF) {
            self.follow = true;
        }
        (forward, right)
    }

    /// One fixed step of game logic.
    pub fn update(&mut self, ctx: &mut Context, dt: f32) {
        let (forward, right) = self.camera(ctx, dt);
        let input = ctx.input;

        // Player: WASD, relative to the camera.
        let player_position = {
            let mut transform = self.world.get::<&mut Transform>(self.player).unwrap();
            let wish = forward * input.axis(KeyCode::KeyS, KeyCode::KeyW)
                + right * input.axis(KeyCode::KeyA, KeyCode::KeyD);
            if wish != Vec3::ZERO {
                let step = wish.normalize() * PLAYER_SPEED * dt;
                let limit = terrain::SIZE / 2.0 - 1.0;
                let mut to = vec2(transform.position.x + step.x, transform.position.z + step.z)
                    .clamp(Vec2::splat(-limit), Vec2::splat(limit));
                // Slide round rocks instead of through them.
                for &(centre, radius) in &self.obstacles {
                    let away = to - centre;
                    if away.length() < radius + 0.5 {
                        to = centre + away.normalize_or(Vec2::X) * (radius + 0.5);
                    }
                }
                transform.position = on_ground(to.x, to.y, 0.5);
                transform.rotation = Quat::from_rotation_y(step.x.atan2(step.z));
            }
            transform.position
        };
        if self.follow {
            // Frame-rate independent smoothing towards the player.
            let blend = 1.0 - (-8.0 * dt).exp();
            ctx.orbit.target = ctx.orbit.target.lerp(player_position, blend);
        }

        // Click: build a tower on the cell under the cursor.
        if input.clicked {
            let size = glam::vec2(ctx.gpu.config.width as f32, ctx.gpu.config.height as f32);
            let (origin, direction) = ctx.orbit.ray(input.cursor, size);
            let cell = terrain::raycast(origin, direction).map(Grid::cell_of);
            match cell {
                Some(cell) if self.grid.is_free(cell) && cell != Grid::cell_of(player_position) => {
                    let (x, z) = Grid::center(cell);
                    self.world.spawn((
                        Tower,
                        Name("Tower".into()),
                        Transform { scale: vec3(0.9, 2.0, 0.9), ..Transform::at(on_ground(x, z, 0.9)) },
                        Renderable {
                            mesh: self.cube.clone(),
                            texture: self.white.clone(),
                            tint: Vec4::new(0.6, 0.6, 0.65, 1.0),
                        },
                    ));
                    self.grid.block(cell);
                    // Nothing grows under a tower.
                    ctx.renderer.grass.paint(vec3(x, 0.0, z), TOWER_CLEARING, -1.0);
                    self.repath();
                    ctx.audio.beep(660.0);
                }
                _ => ctx.audio.beep(180.0),
            }
        }

        if std::mem::take(&mut self.clear_towers) {
            let towers: Vec<_> = self
                .world
                .query::<(hecs::Entity, &Tower, &Transform)>()
                .iter()
                .map(|(entity, _, transform)| (entity, transform.position))
                .collect();
            for (tower, position) in towers {
                self.world.despawn(tower).unwrap();
                ctx.renderer.grass.paint(position, TOWER_CLEARING, 1.0);
            }
            self.grid.clear();
            self.block_obstacles();
            self.repath();
        }

        if std::mem::take(&mut self.spawn_fox) {
            self.spawn_walker(ctx, Grid::cell_of(player_position));
        }

        // Walkers: head for the next cell on the path; leave on arrival.
        let mut arrived = Vec::new();
        for (entity, transform, walker) in
            self.world.query_mut::<(hecs::Entity, &mut Transform, &mut Walker)>()
        {
            let Some(&cell) = walker.path.get(walker.next) else {
                arrived.push(entity);
                continue;
            };
            let (x, z) = Grid::center(cell);
            let to_cell = vec3(x - transform.position.x, 0.0, z - transform.position.z);
            let step = WALKER_SPEED * dt;
            if to_cell.length() <= step {
                walker.next += 1;
            } else {
                let step = to_cell.normalize() * step;
                let (x, z) = (transform.position.x + step.x, transform.position.z + step.z);
                transform.position = on_ground(x, z, 0.0);
                transform.rotation = Quat::from_rotation_y(step.x.atan2(step.z));
            }
        }
        for entity in arrived {
            self.world.despawn(entity).unwrap();
            ctx.audio.beep(440.0);
        }
    }

    /// An animated fox that runs from a corner of the field to `goal`.
    fn spawn_walker(&mut self, ctx: &Context, goal: Cell) {
        let Some(fox) = &self.fox else { return };
        let Some(skeleton) = &fox.skeleton else { return };
        let start = (goal.0 - 30, goal.1 - 30);
        let Some(path) = self.grid.path(start, goal) else {
            ctx.audio.beep(180.0);
            return;
        };
        let (x, z) = Grid::center(start);
        let clip = skeleton.clip_named("Run").unwrap_or(0);
        self.world.spawn((
            Walker { goal, path, next: 0 },
            Name("Fox".into()),
            GrassPusher(1.3),
            // The fox is modelled about 100 units long.
            Transform { scale: Vec3::splat(0.02), ..Transform::at(on_ground(x, z, 0.0)) },
            Renderable { mesh: fox.mesh.clone(), texture: fox.texture.clone(), tint: Vec4::ONE },
            Animator::new(&ctx.gpu.device, &ctx.renderer.joints_layout, skeleton.clone(), clip),
        ));
    }

    /// The set of blocked cells changed: every walker plans again from where
    /// it stands. One with no way through stops and leaves.
    fn repath(&mut self) {
        for (transform, walker) in self.world.query_mut::<(&Transform, &mut Walker)>() {
            let here = Grid::cell_of(transform.position);
            walker.path = self.grid.path(here, walker.goal).unwrap_or_default();
            walker.next = 0;
        }
    }

    pub fn ui(&mut self, context: &egui::Context) {
        egui::Window::new("Sandbox").default_pos([190.0, 44.0]).show(context, |ui| {
            ui.checkbox(&mut self.follow, "Camera follows cube (F)");
            ui.add_enabled_ui(self.fox.is_some(), |ui| {
                self.spawn_fox |= ui.button("Spawn fox").clicked();
            });
            self.clear_towers |= ui.button("Clear towers").clicked();
            ui.separator();
            ui.small("WASD move - click build - right-drag / Q E turn\nscroll zoom - arrows pan");
        });
    }
}

/// A point on the field's surface, raised by `lift`.
fn on_ground(x: f32, z: f32, lift: f32) -> Vec3 {
    vec3(x, terrain::height(x, z) + lift, z)
}
