use crate::app::gpu::Gpu;
use winit::window::Window;

/// egui, wired to the window's events and drawn over the finished 3D frame.
pub struct Ui {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    frame: Option<(Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32)>,
}

impl Ui {
    pub fn new(gpu: &Gpu, window: &Window) -> Self {
        let context = egui::Context::default();
        let state = egui_winit::State::new(
            context,
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, Default::default());
        Self { state, renderer, frame: None }
    }

    /// Feeds a window event to egui. True means egui used it (a click on a
    /// panel, typing in a text box) and the game should ignore it.
    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Builds this frame's UI. `build` gets the root area to dock panels in.
    pub fn run(&mut self, window: &Window, build: impl FnMut(&mut egui::Ui)) {
        let input = self.state.take_egui_input(window);
        let context = self.state.egui_ctx().clone();
        let output = context.run_ui(input, build);
        self.state.handle_platform_output(window, output.platform_output);
        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        self.frame = Some((primitives, output.textures_delta, output.pixels_per_point));
    }

    /// Draws what the last `run` built straight onto `target`.
    pub fn paint(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let Some((primitives, mut textures, pixels_per_point)) = self.frame.take() else { return };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point,
        };

        for (id, deltas) in &textures.set {
            for delta in deltas {
                self.renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }
        self.renderer.update_buffers(&gpu.device, &gpu.queue, encoder, &primitives, &screen);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("UI Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &primitives, &screen);
        }

        for id in &textures.free {
            self.renderer.free_texture(id);
        }
        // egui asserts that a delta is never dropped unapplied.
        textures.clear();
    }
}
