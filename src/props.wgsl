// Props: plants and rocks. Many copies of a few vertex-coloured meshes, each
// copy placed by one instance. Plants bend in the wind; rocks don't.

struct Camera {
  view_proj: mat4x4<f32>,
  eye: vec4<f32>,
}

// Same as in shader.wgsl.
struct Light {
  direction: vec4<f32>,
  color: vec4<f32>,
  params: vec4<f32>,
  view_proj: mat4x4<f32>,
}

struct Air {
  wind: vec4<f32>,  // x, y: direction on the ground (xz), z: strength, w: speed
  time: vec4<f32>,  // x: seconds
}

@group(1) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(0) var<uniform> light: Light;
@group(2) @binding(1) var t_shadow: texture_depth_2d;
@group(2) @binding(2) var s_shadow: sampler_comparison;
@group(3) @binding(0) var<uniform> air: Air;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  // How far this vertex moves in the wind: 0 at the root, most at the top.
  @location(3) sway: f32,
  @location(4) place: vec4<f32>,  // xyz: where the prop stands, w: its scale
  @location(5) turn: vec4<f32>,   // x: rotation about Y, yzw: tint
}

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
}

fn turned(v: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

fn world_position(in: VertexInput) -> vec3<f32> {
  var position = in.place.xyz + turned(in.position, in.turn.x) * in.place.w;

  // Lean with the wind, rocking slowly, with a quicker flutter across it.
  // Neighbours move out of step because the phase depends on where they stand.
  let along = vec3<f32>(air.wind.x, 0.0, air.wind.y);
  let across = vec3<f32>(-air.wind.y, 0.0, air.wind.x);
  let phase = in.place.x * 0.31 + in.place.z * 0.23;
  let t = air.time.x * (1.0 + air.wind.w);
  let lean = 0.12 + 0.1 * sin(t * 0.7 + phase);
  let flutter = 0.04 * sin(t * 2.3 + phase * 1.7 + in.position.y * 4.0);
  position += (along * lean + across * flutter) * in.sway * in.place.w * air.wind.z;
  return position;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.world_position = world_position(in);
  out.clip_position = camera.view_proj * vec4<f32>(out.world_position, 1.0);
  out.normal = turned(in.normal, in.turn.x);
  out.color = in.color * in.turn.yzw;
  return out;
}

// The shadow pass binds the light's matrix where the camera normally goes.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
  return camera.view_proj * vec4<f32>(world_position(in), 1.0);
}

// 1.0 = fully lit, 0.0 = fully shadowed. Same as in shader.wgsl.
fn shadow(world_position: vec3<f32>) -> f32 {
  let clip = light.view_proj * vec4<f32>(world_position, 1.0);
  let ndc = clip.xyz / clip.w;
  let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
  if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 {
    return 1.0;
  }

  let texel = 1.0 / vec2<f32>(textureDimensions(t_shadow));
  var lit = 0.0;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let offset = vec2<f32>(f32(x), f32(y)) * texel;
      lit += textureSampleCompareLevel(t_shadow, s_shadow, uv + offset, ndc.z - light.params.z);
    }
  }
  // Shadows fade out towards the edge of the area the shadow map covers,
  // instead of stopping at a line.
  let edge = max(abs(ndc.x), abs(ndc.y));
  return mix(lit / 9.0, 1.0, smoothstep(0.75, 1.0, edge));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let view = normalize(camera.eye.xyz - in.world_position);
  let to_light = normalize(-light.direction.xyz);

  // Leaves and petals are seen from both sides: light the side looked at.
  var normal = normalize(in.normal);
  if dot(normal, view) < 0.0 {
    normal = -normal;
  }

  // Light wraps a little past the edge, which keeps petals soft while
  // leaving the facets of a rock distinct.
  let diffuse = max((dot(normal, to_light) + 0.25) / 1.25, 0.0) * shadow(in.world_position);
  // Brighter from the sky than from the ground.
  let ambient = light.params.x * mix(0.65, 1.2, normal.y * 0.5 + 0.5);
  return vec4<f32>(in.color * light.color.rgb * (ambient + diffuse), 1.0);
}
