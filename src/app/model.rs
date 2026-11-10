//! glTF loading and skeletal animation.

use crate::app::mesh::Mesh;
use crate::app::texture::Texture;
use crate::app::vertex::Vertex;
use crate::error::{AppError, Result};
use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;

/// Size of the joint array in shader.wgsl.
// ponytail: fixed uniform per entity; use a shared storage buffer for bigger rigs or crowds.
pub const MAX_JOINTS: usize = 64;

struct Node {
    parent: Option<usize>,
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

enum Values {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
}

struct Channel {
    node: usize,
    times: Vec<f32>,
    values: Values,
}

pub struct Clip {
    pub name: String,
    pub duration: f32,
    channels: Vec<Channel>,
}

/// A rig and the animations that drive it.
pub struct Skeleton {
    nodes: Vec<Node>,
    /// Node indices, parents before children.
    order: Vec<usize>,
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
    pub clips: Vec<Clip>,
}

impl Skeleton {
    pub fn clip_named(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|clip| clip.name == name)
    }

    /// One matrix per joint for `clip` at `time` seconds (looping). An
    /// out-of-range clip gives the rest pose.
    pub fn pose(&self, clip: usize, time: f32) -> Vec<Mat4> {
        let mut local: Vec<(Vec3, Quat, Vec3)> =
            self.nodes.iter().map(|n| (n.scale, n.rotation, n.translation)).collect();

        if let Some(clip) = self.clips.get(clip) {
            let time = if clip.duration > 0.0 { time.rem_euclid(clip.duration) } else { 0.0 };
            for channel in &clip.channels {
                let (a, b, t) = keyframes(&channel.times, time);
                let (scale, rotation, translation) = &mut local[channel.node];
                match &channel.values {
                    Values::Translation(v) => *translation = v[a].lerp(v[b], t),
                    Values::Rotation(v) => *rotation = v[a].slerp(v[b], t),
                    Values::Scale(v) => *scale = v[a].lerp(v[b], t),
                }
            }
        }

        let mut world = vec![Mat4::IDENTITY; self.nodes.len()];
        for &index in &self.order {
            let (scale, rotation, translation) = local[index];
            let matrix = Mat4::from_scale_rotation_translation(scale, rotation, translation);
            world[index] = match self.nodes[index].parent {
                Some(parent) => world[parent] * matrix,
                None => matrix,
            };
        }

        self.joints.iter().zip(&self.inverse_bind).map(|(&joint, &bind)| world[joint] * bind).collect()
    }
}

/// The two keyframes around `time` and how far between them it is.
fn keyframes(times: &[f32], time: f32) -> (usize, usize, f32) {
    let next = times.partition_point(|&t| t <= time);
    if next == 0 {
        return (0, 0, 0.0);
    }
    if next == times.len() {
        return (next - 1, next - 1, 0.0);
    }
    let (a, b) = (times[next - 1], times[next]);
    (next - 1, next, (time - a) / (b - a))
}

/// Component: plays a clip on a skeleton and owns the GPU buffer the posed
/// joints are uploaded to.
pub struct Animator {
    pub skeleton: Arc<Skeleton>,
    pub clip: usize,
    pub time: f32,
    buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Animator {
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Joints Bind Layout"),
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
        })
    }

    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        skeleton: Arc<Skeleton>,
        clip: usize,
    ) -> Self {
        let (buffer, bind_group) = joint_buffer(device, layout);
        Self { skeleton, clip, time: 0.0, buffer, bind_group }
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        let pose: Vec<[f32; 16]> =
            self.skeleton.pose(self.clip, self.time).iter().map(Mat4::to_cols_array).collect();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&pose));
    }
}

/// A zeroed joint array and its bind group. Unskinned draws bind one of these
/// too, since the pipeline layout always has the slot.
pub fn joint_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Joints Buffer"),
        size: (MAX_JOINTS * size_of::<Mat4>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Joints Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
    });
    (buffer, bind_group)
}

/// Advances every animator. Runs once per rendered frame, not per fixed step,
/// so animation stays smooth at any refresh rate.
pub fn animate(world: &mut hecs::World, dt: f32) {
    for animator in world.query_mut::<&mut Animator>() {
        animator.time += dt;
    }
}

/// A glTF file decoded on the CPU.
pub struct ModelData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Base colour as RGBA8: width, height, pixels.
    pub image: (u32, u32, Vec<u8>),
    pub skeleton: Option<Skeleton>,
}

/// A glTF file on the GPU, ready to hand to entities.
pub struct Model {
    pub mesh: Arc<Mesh>,
    pub texture: Arc<Texture>,
    pub skeleton: Option<Arc<Skeleton>>,
}

impl Model {
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_layout: &wgpu::BindGroupLayout,
        path: &str,
    ) -> Result<Self> {
        let data = ModelData::load(path)?;
        let (width, height, pixels) = &data.image;
        Ok(Self {
            mesh: Arc::new(Mesh::new(device, path, &data.vertices, &data.indices)),
            texture: Arc::new(Texture::from_rgba(
                device, queue, texture_layout, path, *width, *height, pixels,
            )),
            skeleton: data.skeleton.map(Arc::new),
        })
    }
}

impl ModelData {
    // ponytail: all primitives merge into one mesh with the first material and skin.
    pub fn load(path: &str) -> Result<Self> {
        let unsupported = |reason: &str| AppError::ModelUnsupported {
            path: path.to_string(),
            reason: reason.to_string(),
        };
        let (document, buffers, images) = gltf::import(path)
            .map_err(|source| AppError::LoadModel { path: path.to_string(), source })?;
        let buffer = |b: gltf::Buffer| Some(&buffers[b.index()].0[..]);

        // World matrix and parent of every node.
        let mut parents = vec![None; document.nodes().len()];
        let mut world = vec![Mat4::IDENTITY; document.nodes().len()];
        let mut order = Vec::new();
        let mut stack: Vec<(gltf::Node, Option<usize>)> =
            document.scenes().flat_map(|s| s.nodes()).map(|n| (n, None)).collect();
        while let Some((node, parent)) = stack.pop() {
            let matrix = Mat4::from_cols_array_2d(&node.transform().matrix());
            world[node.index()] = parent.map_or(matrix, |p: usize| world[p] * matrix);
            parents[node.index()] = parent;
            order.push(node.index());
            stack.extend(node.children().map(|child| (child, Some(node.index()))));
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut image = None;
        for node in document.nodes() {
            let Some(mesh) = node.mesh() else { continue };
            // Skinned vertices are placed by their joints; rigid ones by the node.
            let skinned = node.skin().is_some();
            let place = if skinned { Mat4::IDENTITY } else { world[node.index()] };

            for primitive in mesh.primitives() {
                let reader = primitive.reader(buffer);
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or_else(|| unsupported("a primitive has no positions"))?
                    .collect();
                let count = positions.len();
                let normals: Vec<[f32; 3]> = match reader.read_normals() {
                    Some(normals) => normals.collect(),
                    None => vec![[0.0, 1.0, 0.0]; count],
                };
                let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                    Some(uvs) => uvs.into_f32().collect(),
                    None => vec![[0.0; 2]; count],
                };
                let joints: Vec<[u16; 4]> = match reader.read_joints(0) {
                    Some(joints) if skinned => joints.into_u16().collect(),
                    _ => vec![[0; 4]; count],
                };
                let weights: Vec<[f32; 4]> = match reader.read_weights(0) {
                    Some(weights) if skinned => weights.into_f32().collect(),
                    _ => vec![[0.0; 4]; count],
                };

                let base = vertices.len() as u32;
                match reader.read_indices() {
                    Some(read) => indices.extend(read.into_u32().map(|i| base + i)),
                    None => indices.extend(base..base + count as u32),
                }
                vertices.extend((0..count).map(|i| Vertex {
                    position: place.transform_point3(positions[i].into()).to_array(),
                    uv_coord: uvs[i],
                    normal: place.transform_vector3(normals[i].into()).normalize_or_zero().to_array(),
                    joints: joints[i].map(u32::from),
                    weights: weights[i],
                }));

                if image.is_none() {
                    let pbr = primitive.material().pbr_metallic_roughness();
                    image = Some(match pbr.base_color_texture() {
                        Some(info) => rgba8(&images[info.texture().source().index()])
                            .ok_or_else(|| unsupported("base colour texture is not 8-bit RGB(A)"))?,
                        // No texture: a single pixel of the material colour
                        // (linear in the file, sRGB in the texture).
                        None => {
                            let srgb = pbr.base_color_factor().map(|c| (c.powf(1.0 / 2.2) * 255.0) as u8);
                            (1, 1, vec![srgb[0], srgb[1], srgb[2], 255])
                        }
                    });
                }
            }
        }
        if vertices.is_empty() {
            return Err(unsupported("no meshes"));
        }

        let skeleton = match document.skins().next() {
            None => None,
            Some(skin) => {
                let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
                if joints.len() > MAX_JOINTS {
                    return Err(unsupported("more joints than MAX_JOINTS"));
                }
                let inverse_bind = match skin.reader(buffer).read_inverse_bind_matrices() {
                    Some(matrices) => matrices.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
                    None => vec![Mat4::IDENTITY; joints.len()],
                };
                let nodes = document
                    .nodes()
                    .map(|node| {
                        let (translation, rotation, scale) = node.transform().decomposed();
                        Node {
                            parent: parents[node.index()],
                            translation: translation.into(),
                            rotation: Quat::from_array(rotation),
                            scale: scale.into(),
                        }
                    })
                    .collect();
                let clips = document.animations().map(|a| read_clip(&a, buffer)).collect();
                Some(Skeleton { nodes, order, joints, inverse_bind, clips })
            }
        };

        log::info!(
            "loaded model {path} ({} vertices, {} joints, {} clips)",
            vertices.len(),
            skeleton.as_ref().map_or(0, |s| s.joints.len()),
            skeleton.as_ref().map_or(0, |s| s.clips.len()),
        );
        Ok(Self { vertices, indices, image: image.expect("set with the first primitive"), skeleton })
    }
}

fn read_clip<'a>(
    animation: &gltf::Animation<'a>,
    buffer: impl Fn(gltf::Buffer<'a>) -> Option<&'a [u8]> + Clone,
) -> Clip {
    use gltf::animation::Interpolation;
    use gltf::animation::util::ReadOutputs;

    let mut channels = Vec::new();
    for channel in animation.channels() {
        let reader = channel.reader(buffer.clone());
        let (Some(times), Some(outputs)) = (reader.read_inputs(), reader.read_outputs()) else {
            continue;
        };
        let times: Vec<f32> = times.collect();
        // ponytail: cubic-spline tracks are played as linear through their
        // keyframe values (each key is stored as in-tangent, value, out-tangent).
        let cubic = channel.sampler().interpolation() == Interpolation::CubicSpline;
        fn keys<T>(values: impl Iterator<Item = T>, cubic: bool) -> Vec<T> {
            if cubic { values.skip(1).step_by(3).collect() } else { values.collect() }
        }
        let values = match outputs {
            ReadOutputs::Translations(v) => Values::Translation(keys(v.map(Vec3::from), cubic)),
            ReadOutputs::Rotations(v) => {
                Values::Rotation(keys(v.into_f32().map(Quat::from_array), cubic))
            }
            ReadOutputs::Scales(v) => Values::Scale(keys(v.map(Vec3::from), cubic)),
            ReadOutputs::MorphTargetWeights(_) => continue,
        };
        if !times.is_empty() {
            channels.push(Channel { node: channel.target().node().index(), times, values });
        }
    }

    let duration =
        channels.iter().filter_map(|c| c.times.last().copied()).fold(0.0, f32::max);
    Clip { name: animation.name().unwrap_or_default().to_string(), duration, channels }
}

fn rgba8(image: &gltf::image::Data) -> Option<(u32, u32, Vec<u8>)> {
    use gltf::image::Format;
    let pixels = match image.format {
        Format::R8G8B8A8 => image.pixels.clone(),
        Format::R8G8B8 => image.pixels.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        _ => return None,
    };
    Some((image.width, image.height, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bone(parent: Option<usize>, y: f32) -> Node {
        Node { parent, translation: Vec3::Y * y, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }

    #[test]
    fn rest_pose_is_identity_and_rotation_carries_children() {
        // Two bones stacked on Y; the root's animation turns it 90 degrees
        // about Z over one second.
        let quarter = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let skeleton = Skeleton {
            nodes: vec![bone(None, 0.0), bone(Some(0), 1.0)],
            order: vec![0, 1],
            joints: vec![0, 1],
            inverse_bind: vec![Mat4::IDENTITY, Mat4::from_translation(-Vec3::Y)],
            clips: vec![Clip {
                name: "turn".into(),
                duration: 2.0,
                channels: vec![Channel {
                    node: 0,
                    times: vec![0.0, 1.0, 2.0],
                    values: Values::Rotation(vec![Quat::IDENTITY, quarter, quarter]),
                }],
            }],
        };

        for joint in skeleton.pose(99, 0.0) {
            assert!(joint.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }

        // A vertex at the tip of the child bone swings from +Y to -X.
        let tip = Vec3::Y * 2.0;
        let child = |time| skeleton.pose(0, time)[1].transform_point3(tip);
        assert!(child(1.0).abs_diff_eq(Vec3::X * -2.0, 1e-4), "{}", child(1.0));
        // Halfway there it is 45 degrees over; and time loops.
        let half = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4) * tip;
        assert!(child(0.5).abs_diff_eq(half, 1e-4));
        assert!(child(2.5).abs_diff_eq(half, 1e-4));
    }

    #[test]
    fn fox_loads_with_rig_and_clips() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/Fox.glb");
        let fox = ModelData::load(path).unwrap();
        let skeleton = fox.skeleton.expect("fox is rigged");
        assert_eq!(skeleton.joints.len(), 24);
        assert!(skeleton.clip_named("Run").is_some());
        assert!(fox.vertices.iter().all(|v| v.weights.iter().sum::<f32>() > 0.99));

        // The rest pose leaves the mesh where it was modelled (bind pose), and
        // an animated pose stays finite.
        for joint in skeleton.pose(usize::MAX, 0.0) {
            assert!(joint.abs_diff_eq(Mat4::IDENTITY, 1e-3), "{joint}");
        }
        assert!(skeleton.pose(0, 0.4).iter().all(|joint| joint.is_finite()));
    }
}
