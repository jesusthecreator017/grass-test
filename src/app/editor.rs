//! The in-game editor: a toolbar to pause and step the game, a list of every
//! entity, and an inspector for the selected one. F1 hides it.

use crate::app::grass::{FAR_CAPACITY, GrassSettings, NEAR_CAPACITY, TYPE_NAMES};
use crate::app::model::Animator;
use crate::app::renderer::{Name, Renderable, Renderer, Transform};
use egui::{DragValue, Panel, Slider};
use glam::{EulerRot, Quat, Vec3};
use serde::{Deserialize, Serialize};

const SETTINGS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/editor.json");
const DENSITY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/grass_density.png");

/// What the Save button writes: everything tuned in the editor's panels.
/// The painted grass density goes next to it as an image.
#[derive(Serialize, Deserialize)]
struct Saved {
    grass: GrassSettings,
    sun_direction: [f32; 3],
    sun_color: [f32; 3],
    ambient: f32,
    specular: f32,
    shadow_bias: f32,
    sky: [f32; 3],
}

pub fn save(renderer: &Renderer) -> Result<(), String> {
    let light = &renderer.light;
    let saved = Saved {
        grass: renderer.grass.settings.clone(),
        sun_direction: light.direction.to_array(),
        sun_color: light.color.to_array(),
        ambient: light.ambient,
        specular: light.specular,
        shadow_bias: light.shadow_bias,
        sky: renderer.sky,
    };
    let json = serde_json::to_string_pretty(&saved).map_err(|err| err.to_string())?;
    std::fs::write(SETTINGS_PATH, json).map_err(|err| err.to_string())?;
    renderer.grass.save_density(DENSITY_PATH)
}

/// Applies what `save` wrote. Nothing saved yet leaves the defaults; a file
/// that can't be read is reported and ignored.
pub fn load(renderer: &mut Renderer) {
    if let Ok(json) = std::fs::read_to_string(SETTINGS_PATH) {
        match serde_json::from_str::<Saved>(&json) {
            Ok(saved) => {
                renderer.grass.settings = saved.grass;
                let light = &mut renderer.light;
                light.direction =
                    Vec3::from(saved.sun_direction).try_normalize().unwrap_or(light.direction);
                light.color = saved.sun_color.into();
                light.ambient = saved.ambient;
                light.specular = saved.specular;
                light.shadow_bias = saved.shadow_bias;
                renderer.sky = saved.sky;
                log::info!("loaded {SETTINGS_PATH}");
            }
            Err(err) => log::warn!("ignoring {SETTINGS_PATH}: {err}"),
        }
    }
    renderer.grass.load_density(DENSITY_PATH);
}

pub struct Editor {
    pub visible: bool,
    /// False = edit mode: the game is frozen and clicks select entities.
    pub playing: bool,
    /// Run exactly one update while paused.
    pub step: bool,
    pub selected: Option<hecs::Entity>,
    /// Edit mode tool: dragging on the ground brushes grass instead of
    /// clicking selecting.
    pub painting: bool,
    pub erasing: bool,
    pub brush_radius: f32,
    /// Outcome of the last Save.
    status: String,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            visible: true,
            playing: true,
            step: false,
            selected: None,
            painting: false,
            erasing: true,
            brush_radius: 4.0,
            status: String::new(),
        }
    }
}

impl Editor {
    pub fn ui(&mut self, ui: &mut egui::Ui, world: &hecs::World, renderer: &mut Renderer, stats: &str) {
        // Entities come and go while the game runs.
        self.selected = self.selected.filter(|&entity| world.contains(entity));
        if !self.visible {
            return;
        }

        Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                let label = if self.playing { "Pause" } else { "Play" };
                if ui.button(label).clicked() {
                    self.playing = !self.playing;
                }
                self.step |= ui.add_enabled(!self.playing, egui::Button::new("Step")).clicked();
                if ui.button("Save").on_hover_text("lighting, grass settings, painted grass").clicked() {
                    self.status = match save(renderer) {
                        Ok(()) => "saved".into(),
                        Err(err) => format!("save failed: {err}"),
                    };
                }
                ui.weak(&self.status);
                ui.separator();
                ui.label(match (self.playing, self.painting) {
                    (true, _) => "playing",
                    (false, true) => "editing - drag paints grass",
                    (false, false) => "editing - click selects",
                });
                ui.separator();
                ui.label(format!("{stats}   {} entities   {} props", world.len(), renderer.props.count));
                if let Some(error) = &renderer.shader_error {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, "shader error (see inspector)")
                        .on_hover_text(error);
                }
            });
        });

        Panel::left("hierarchy").default_size(170.0).show(ui, |ui| {
            ui.heading("Entities");
            let mut rows: Vec<(hecs::Entity, String)> = world
                .query::<(hecs::Entity, Option<&Name>)>()
                .iter()
                .map(|(entity, name)| {
                    let name = name.map_or("(unnamed)", |n| n.0.as_str());
                    (entity, format!("{name} #{}", entity.id()))
                })
                .collect();
            rows.sort_by_key(|row| row.0.id());
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                for (entity, label) in rows {
                    if ui.selectable_label(self.selected == Some(entity), label).clicked() {
                        self.selected = Some(entity);
                    }
                }
            });
        });

        Panel::right("inspector").default_size(270.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                ui.heading("Inspector");
                match self.selected {
                    Some(entity) => inspect(ui, world, entity),
                    None => {
                        ui.weak("Nothing selected. Pick from the list, or pause and click.");
                    }
                }
                ui.separator();
                ui.collapsing("Lighting", |ui| lighting(ui, renderer));
                egui::CollapsingHeader::new("Grass").default_open(true).show(ui, |ui| {
                    ui.checkbox(&mut self.painting, "Paint tool (while paused)");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.erasing, true, "erase");
                        ui.radio_value(&mut self.erasing, false, "grow");
                    });
                    ui.add(Slider::new(&mut self.brush_radius, 1.0..=20.0).text("brush radius"));
                    ui.separator();
                    let [near, far] = renderer.grass.blade_counts;
                    ui.label(format!("{near} near + {far} far blades"));
                    if near as u64 > NEAR_CAPACITY || far as u64 > FAR_CAPACITY {
                        ui.colored_label(
                            egui::Color32::LIGHT_RED,
                            "More blades than fit: some are missing. Raise blade spacing or lower a radius.",
                        );
                    }
                    grass(ui, &mut renderer.grass.settings);
                });
                if let Some(error) = &renderer.shader_error {
                    ui.separator();
                    ui.heading("Shader error");
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            });
        });
    }
}

/// The entity under a world-space ray: whichever mesh it hits first.
pub fn pick(world: &hecs::World, origin: Vec3, direction: Vec3) -> Option<hecs::Entity> {
    world
        .query::<(hecs::Entity, &Transform, &Renderable)>()
        .iter()
        .filter_map(|(entity, transform, renderable)| {
            // Ray goes into mesh space; unnormalised direction keeps distances comparable.
            let to_local = transform.matrix().inverse();
            let distance = renderable
                .mesh
                .raycast(to_local.transform_point3(origin), to_local.transform_vector3(direction))?;
            Some((entity, distance))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(entity, _)| entity)
}

fn inspect(ui: &mut egui::Ui, world: &hecs::World, entity: hecs::Entity) {
    if let Ok(mut name) = world.get::<&mut Name>(entity) {
        ui.text_edit_singleline(&mut name.0);
    }

    if let Ok(mut transform) = world.get::<&mut Transform>(entity) {
        ui.label("Transform");
        drag3(ui, "position", &mut transform.position, 0.05);

        // Shown as yaw / pitch / roll in degrees; stored as a quaternion.
        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let mut degrees = Vec3::new(pitch, yaw, roll).map(f32::to_degrees);
        if drag3(ui, "rotation", &mut degrees, 1.0) {
            let radians = degrees.map(f32::to_radians);
            transform.rotation = Quat::from_euler(EulerRot::YXZ, radians.y, radians.x, radians.z);
        }
        drag3(ui, "scale", &mut transform.scale, 0.01);
    }

    if let Ok(mut renderable) = world.get::<&mut Renderable>(entity) {
        ui.horizontal(|ui| {
            let mut tint = renderable.tint.to_array();
            if ui.color_edit_button_rgba_unmultiplied(&mut tint).changed() {
                renderable.tint = tint.into();
            }
            ui.label("tint");
        });
    }

    if let Ok(mut animator) = world.get::<&mut Animator>(entity) {
        let skeleton = animator.skeleton.clone();
        ui.label("Animation");
        let current = skeleton.clips.get(animator.clip).map_or("(rest pose)", |c| c.name.as_str());
        egui::ComboBox::from_id_salt("clip").selected_text(current).show_ui(ui, |ui| {
            for (index, clip) in skeleton.clips.iter().enumerate() {
                ui.selectable_value(&mut animator.clip, index, clip.name.as_str());
            }
        });
        if let Some(clip) = skeleton.clips.get(animator.clip) {
            // Scrub while paused; while playing it just shows progress.
            let mut time = animator.time.rem_euclid(clip.duration.max(f32::EPSILON));
            if ui.add(Slider::new(&mut time, 0.0..=clip.duration).text("time")).changed() {
                animator.time = time;
            }
        }
    }
}

fn lighting(ui: &mut egui::Ui, renderer: &mut Renderer) {
    let light = &mut renderer.light;

    // The sun as compass bearing and height above the horizon. Capped below
    // straight overhead, where "which way is up" stops having an answer.
    let towards_sun = -light.direction;
    let mut azimuth = towards_sun.x.atan2(towards_sun.z).to_degrees();
    let mut elevation = towards_sun.y.clamp(-1.0, 1.0).asin().to_degrees();
    let changed = ui.add(Slider::new(&mut azimuth, -180.0..=180.0).text("sun azimuth")).changed()
        | ui.add(Slider::new(&mut elevation, 5.0..=85.0).text("sun elevation")).changed();
    if changed {
        let (azimuth, elevation) = (azimuth.to_radians(), elevation.to_radians());
        light.direction = -Vec3::new(
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
            elevation.cos() * azimuth.cos(),
        );
    }

    ui.horizontal(|ui| {
        let mut color = light.color.to_array();
        if ui.color_edit_button_rgb(&mut color).changed() {
            light.color = color.into();
        }
        ui.label("sun colour");
    });
    ui.add(Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"));
    ui.add(Slider::new(&mut light.specular, 0.0..=1.0).text("specular"));
    ui.add(Slider::new(&mut light.shadow_bias, 0.0..=0.01).text("shadow bias"));
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut renderer.sky);
        ui.label("sky");
    });
}

fn grass(ui: &mut egui::Ui, s: &mut GrassSettings) {
    ui.checkbox(&mut s.enabled, "enabled");
    ui.checkbox(&mut s.cast_shadows, "blades cast shadows");
    if ui.button("Reset to defaults").clicked() {
        *s = GrassSettings::default();
    }

    ui.label("Density and distance");
    ui.add(Slider::new(&mut s.density, 0.05..=1.0).text("density"));
    ui.add(Slider::new(&mut s.spacing, 0.04..=0.3).text("blade spacing"));
    ui.add(Slider::new(&mut s.near_radius, 5.0..=60.0).text("detail radius"));
    ui.add(Slider::new(&mut s.far_radius, 20.0..=150.0).text("far radius"));

    ui.label("Blade");
    ui.add(Slider::new(&mut s.height, 0.1..=2.5).text("height"));
    ui.add(Slider::new(&mut s.width, 0.01..=0.2).text("width"));
    ui.add(Slider::new(&mut s.tilt, 0.0..=1.4).text("tilt"));
    ui.add(Slider::new(&mut s.bend, 0.0..=0.6).text("bend"));

    ui.label("Height variation");
    ui.add(Slider::new(&mut s.stragglers, 0.0..=1.0).text("stragglers"));
    ui.add(Slider::new(&mut s.tussocks, 0.0..=1.0).text("tussocks"));
    ui.add(Slider::new(&mut s.swell, 0.0..=0.8).text("swell"));

    ui.label("Types");
    for (index, kind) in s.types.iter_mut().enumerate() {
        ui.collapsing(TYPE_NAMES[index], |ui| {
            // Regular grass grows wherever the others don't.
            if index != 1 {
                ui.add(Slider::new(&mut kind.coverage, 0.0..=1.0).text("coverage"));
            }
            ui.add(Slider::new(&mut kind.density, 0.05..=1.0).text("density"));
            ui.add(Slider::new(&mut kind.height, 0.1..=3.0).text("height x"));
            ui.add(Slider::new(&mut kind.width, 0.3..=3.0).text("width x"));
            ui.add(Slider::new(&mut kind.tilt, 0.0..=2.0).text("tilt x"));
            ui.add(Slider::new(&mut kind.bend, 0.0..=3.0).text("bend x"));
            ui.add(Slider::new(&mut kind.wind, 0.0..=3.0).text("wind x"));
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut kind.color);
                ui.add(Slider::new(&mut kind.color_amount, 0.0..=1.0).text("own colour"));
            });
            if index == 3 {
                ui.add(Slider::new(&mut kind.ear, 0.0..=0.2).text("ear size"));
            }
        });
    }

    ui.label("Clumps");
    ui.add(Slider::new(&mut s.clump_size, 0.3..=6.0).text("size"));
    ui.add(Slider::new(&mut s.clump_height, 0.0..=1.0).text("height variation"));
    ui.add(Slider::new(&mut s.clump_facing, 0.0..=1.0).text("splay"));
    ui.add(Slider::new(&mut s.clump_color, 0.0..=0.6).text("colour variation"));

    ui.label("Patches");
    ui.add(Slider::new(&mut s.patch_size, 3.0..=60.0).text("size"));
    ui.add(Slider::new(&mut s.patch_height, 0.0..=0.9).text("height variation"));
    ui.add(Slider::new(&mut s.patch_thinning, 0.0..=0.9).text("thinning"));
    ui.add(Slider::new(&mut s.dry_amount, 0.0..=1.0).text("dry grass"));
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut s.dry_color);
        ui.label("dry colour");
    });

    ui.label("Flowers");
    ui.add(Slider::new(&mut s.flower_amount, 0.0..=1.0).text("amount"));
    ui.add(Slider::new(&mut s.flower_size, 0.02..=0.3).text("head size"));
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut s.flower_color);
        ui.color_edit_button_rgb(&mut s.flower2_color);
        ui.label("colours");
    });

    ui.label("Ground");
    ui.add(Slider::new(&mut s.path_width, 0.0..=6.0).text("path width"));
    ui.add(Slider::new(&mut s.worn, 0.0..=1.0).text("worn patches"));
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut s.soil_color);
        ui.label("soil");
    });

    ui.label("Colour");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut s.root_color);
        ui.label("root");
        ui.color_edit_button_rgb(&mut s.tip_color);
        ui.label("tip");
    });
    ui.add(Slider::new(&mut s.occlusion, 0.0..=1.0).text("root darkening"));
    ui.add(Slider::new(&mut s.translucency, 0.0..=1.5).text("translucency"));
    ui.add(Slider::new(&mut s.specular, 0.0..=1.0).text("specular"));

    ui.label("Wind");
    ui.add(Slider::new(&mut s.wind_angle, 0.0..=360.0).text("direction"));
    ui.add(Slider::new(&mut s.wind_strength, 0.0..=2.0).text("strength"));
    ui.add(Slider::new(&mut s.wind_speed, 0.0..=4.0).text("speed"));
}

/// Three drag boxes for a vector. True if any of them moved.
fn drag3(ui: &mut egui::Ui, label: &str, value: &mut Vec3, speed: f32) -> bool {
    ui.horizontal(|ui| {
        let changed = ui.add(DragValue::new(&mut value.x).speed(speed)).changed()
            | ui.add(DragValue::new(&mut value.y).speed(speed)).changed()
            | ui.add(DragValue::new(&mut value.z).speed(speed)).changed();
        ui.label(label);
        changed
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_angles_round_trip() {
        // The same maths the lighting panel uses, there and back.
        let direction = Vec3::new(-0.6, -1.0, -0.4).normalize();
        let sun = -direction;
        let (azimuth, elevation) = (sun.x.atan2(sun.z), sun.y.asin());
        let back = -Vec3::new(
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
            elevation.cos() * azimuth.cos(),
        );
        assert!(back.abs_diff_eq(direction, 1e-5), "{back} vs {direction}");
    }
}
