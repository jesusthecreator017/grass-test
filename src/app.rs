pub mod audio;
pub mod camera;
mod editor;
mod frame_stats;
pub mod grass;
pub mod gpu;
pub mod input;
mod instance;
mod light;
pub mod mesh;
pub mod model;
pub mod nav;
pub mod props;
pub mod renderer;
pub mod terrain;
pub mod texture;
mod ui;
pub mod vertex;

use crate::app::audio::Audio;
use crate::app::camera::Orbit;
use crate::app::editor::Editor;
use crate::app::frame_stats::FrameStats;
use crate::app::gpu::Gpu;
use crate::app::input::Input;
use crate::app::renderer::Renderer;
use crate::app::ui::Ui;
use crate::error::{AppError, Result};
use crate::game::{Context, Game};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

/// Game logic runs at this fixed rate no matter how fast frames are drawn.
// ponytail: no interpolation between steps; add if motion judders at high refresh.
const STEP: f32 = 1.0 / 60.0;
/// Longest stretch of real time one frame may try to catch up on.
const MAX_FRAME: f32 = 0.25;
const CAPTURE_FRAME: u32 = 240;

/// What game code gets to touch. A macro, not a method, so the borrow
/// checker can see it leaves `self.game` alone.
macro_rules! context {
    ($state:expr) => {
        Context {
            gpu: &$state.gpu,
            renderer: &mut $state.renderer,
            input: &$state.input,
            orbit: &mut $state.orbit,
            audio: &$state.audio,
        }
    };
}

#[derive(Default)]
pub struct App {
    state: Option<State>,
    error: Option<AppError>,
}

impl App {
    pub fn take_error(&mut self) -> Option<AppError> {
        self.error.take()
    }
}

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    renderer: Renderer,
    ui: Ui,
    input: Input,
    orbit: Orbit,
    audio: Audio,
    game: Game,
    editor: Editor,
    stats: FrameStats,
    last_frame: Instant,
    /// Real time not yet consumed by fixed steps.
    accumulator: f32,
    frames: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Check if state exist already
        if self.state.is_some() {
            return;
        }

        // Create state
        let attrs = Window::default_attributes()
            .with_title("grass-test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0))
            // A capture run is automated: don't take the keyboard from
            // whoever is typing in another window.
            .with_active(std::env::var_os("CAPTURE").is_none());
        match create_state(event_loop, attrs) {
            Ok(state) => {
                // Needed to immidietly render upon window creation.
                state.window.request_redraw();
                self.state = Some(state);
            }
            Err(err) => {
                self.error = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        // egui sees everything first. Presses it used are not for the game;
        // releases always get through, so nothing is left stuck down.
        let used_by_ui = state.ui.on_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.gpu.resize(size);
                state.window.request_redraw();
            }
            // On window creation Winit throws an occluded, so we need to render on not occluded.
            WindowEvent::Occluded(occluded) => {
                if !occluded {
                    state.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => state.frame(),
            WindowEvent::KeyboardInput { event: key, .. } => {
                if let PhysicalKey::Code(code) = key.physical_key {
                    let down = key.state == ElementState::Pressed;
                    if code == KeyCode::F1 && down && !key.repeat {
                        state.editor.visible = !state.editor.visible;
                    }
                    state.input.key(code, down && !used_by_ui);
                }
            }
            WindowEvent::MouseInput { state: button_state, button, .. } => {
                let down = button_state == ElementState::Pressed && !used_by_ui;
                match button {
                    MouseButton::Left => {
                        state.input.clicked |= down;
                        state.input.left_held = down;
                    }
                    MouseButton::Right => state.orbit.is_dragging(down),
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = glam::vec2(position.x as f32, position.y as f32);
                state.input.cursor = position;
                state.orbit.cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } if !used_by_ui => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => lines,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 * 0.05,
                };
                state.orbit.zoom(amount);
            }
            _ => {}
        }
    }
}

fn create_state(event_loop: &ActiveEventLoop, attrs: WindowAttributes) -> Result<State> {
    let window = Arc::new(event_loop.create_window(attrs)?);
    State::new(window)
}

impl State {
    fn new(window: Arc<Window>) -> Result<Self> {
        let gpu = Gpu::new(window.clone())?;
        let mut renderer = Renderer::new(&gpu);
        editor::load(&mut renderer);
        let ui = Ui::new(&gpu, &window);
        let game = Game::new(&gpu, &mut renderer)?;

        Ok(Self {
            window,
            gpu,
            renderer,
            ui,
            input: Input::default(),
            orbit: Orbit::default(),
            audio: Audio::new(),
            game,
            editor: Editor::default(),
            stats: FrameStats::default(),
            last_frame: Instant::now(),
            accumulator: 0.0,
            frames: 0,
        })
    }

    /// One displayed frame: as many fixed updates as real time calls for,
    /// then a draw.
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(MAX_FRAME);
        self.last_frame = now;

        self.renderer.reload_shaders_if_changed(&self.gpu);

        if self.editor.playing {
            self.accumulator += dt;
            while self.accumulator >= STEP {
                self.accumulator -= STEP;
                self.game.update(&mut context!(self), STEP);
                self.input.end_step();
            }
            model::animate(&mut self.game.world, dt);
        } else {
            // Edit mode: the world holds still, a click selects instead of
            // reaching the game, and the camera still moves.
            self.accumulator = 0.0;
            let clicked = std::mem::take(&mut self.input.clicked);
            let size = glam::vec2(self.gpu.config.width as f32, self.gpu.config.height as f32);
            let (origin, direction) = self.orbit.ray(self.input.cursor, size);
            if self.editor.painting {
                // Hold the button and drag to brush grass in or out.
                if self.input.left_held
                    && let Some(point) = terrain::raycast(origin, direction)
                {
                    let amount = if self.editor.erasing { -1.0 } else { 1.0 } * 4.0 * dt;
                    self.renderer.grass.paint(point, self.editor.brush_radius, amount);
                }
            } else if clicked {
                self.editor.selected = editor::pick(&self.game.world, origin, direction);
            }
            if std::mem::take(&mut self.editor.step) {
                self.game.update(&mut context!(self), STEP);
                model::animate(&mut self.game.world, STEP);
            } else {
                self.game.camera(&mut context!(self), dt);
            }
            self.input.end_step();
        }
        self.renderer.selected = self.editor.selected;

        self.render();
    }

    fn render(&mut self) {
        // Check if the surface texture isnt failing or rezised.
        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                log::warn!("surface outdated or lost, reconfiguring");
                self.gpu.reconfigure();
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                log::warn!("surface acquisition timed out, retrying");
                self.window.request_redraw();
                return;
            }
            // Occluded and Validation: nothing to draw into. Occluded(false)
            // will ask for the next frame.
            other => {
                log::warn!("frame skipped: {other:?}");
                return;
            }
        };

        let stats = self.stats.tick().to_string();
        self.ui.run(&self.window, |ui| {
            self.editor.ui(ui, &self.game.world, &mut self.renderer, &stats);
            if self.editor.visible {
                self.game.ui(ui.ctx());
            }
        });

        // `CAPTURE=shot.png` renders one frame (`CAPTURE_FRAME`) to a file and quits.
        self.frames += 1;
        let capture_frame =
            std::env::var("CAPTURE_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(CAPTURE_FRAME);
        let capture = (self.frames == capture_frame)
            .then(|| std::env::var("CAPTURE").ok())
            .flatten()
            .map(|path| (path, self.gpu.capture_target()));

        // Create the view to draw into and the encoder.
        let view = match &capture {
            Some((_, texture)) => texture.create_view(&Default::default()),
            None => frame.texture.create_view(&Default::default()),
        };
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        self.renderer.render(&self.gpu, &mut encoder, &view, &self.game.world, &self.orbit);
        self.ui.paint(&self.gpu, &mut encoder, &view);

        // Submit the commands into the queue for presentation.
        self.gpu.queue.submit([encoder.finish()]);
        self.renderer.grass.after_submit();
        if let Some((path, texture)) = capture {
            self.gpu.save_png(&texture, &path);
            std::process::exit(0);
        }
        self.gpu.queue.present(frame);
        self.window.request_redraw();
    }
}
