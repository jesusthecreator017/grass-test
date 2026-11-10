struct Camera {
  view_proj: mat4x4<f32>,
  eye: vec4<f32>,
}

struct Light {
  direction: vec4<f32>,
  color: vec4<f32>,
  // x: ambient, y: specular strength, z: extra shadow depth bias
  params: vec4<f32>,
  view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(1) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(0) var<uniform> light: Light;
@group(2) @binding(1) var t_shadow: texture_depth_2d;
@group(2) @binding(2) var s_shadow: sampler_comparison;
// Must match MAX_JOINTS in model.rs.
@group(3) @binding(0) var<uniform> joints: array<mat4x4<f32>, 64>;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
  // One model matrix per instance, arriving as four columns.
  @location(5) model_0: vec4<f32>,
  @location(6) model_1: vec4<f32>,
  @location(7) model_2: vec4<f32>,
  @location(8) model_3: vec4<f32>,
  @location(9) tint: vec4<f32>,
}

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) world_position: vec3<f32>,
  @location(3) tint: vec4<f32>,
}

// Object space to world space. Vertices with no weights are rigid; the rest
// are bent by the skeleton first.
fn world_matrix(in: VertexInput) -> mat4x4<f32> {
  let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
  if in.weights.x + in.weights.y + in.weights.z + in.weights.w == 0.0 {
    return model;
  }
  let skin = joints[in.joints.x] * in.weights.x
    + joints[in.joints.y] * in.weights.y
    + joints[in.joints.z] * in.weights.z
    + joints[in.joints.w] * in.weights.w;
  return model * skin;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  let model = world_matrix(in);
  let world = model * vec4<f32>(in.position, 1.0);

  var out: VertexOutput;
  out.world_position = world.xyz;
  out.clip_position = camera.view_proj * world;
  out.uv = in.uv;
  out.normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
  out.tint = in.tint;
  return out;
}

// The shadow pass binds the light's matrix where the camera normally goes.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
  return camera.view_proj * world_matrix(in) * vec4<f32>(in.position, 1.0);
}

// 1.0 = fully lit, 0.0 = fully shadowed. 3x3 PCF softens the edge.
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
  let base = textureSample(t_diffuse, s_diffuse, in.uv) * in.tint;

  // Interpolation shortens normals, so renormalize them per fragment.
  let normal = normalize(in.normal);
  let to_light = normalize(-light.direction.xyz);
  let to_eye = normalize(camera.eye.xyz - in.world_position);
  let halfway = normalize(to_light + to_eye);

  // Lighting
  let ambient = light.params.x;
  let lit = shadow(in.world_position);
  let diffuse = max(dot(normal, to_light), 0.0) * lit;
  let specular = pow(max(dot(normal, halfway), 0.0), 64.0) * step(0.0001, diffuse);

  let color = base.rgb * light.color.rgb * (ambient + diffuse)
    + light.color.rgb * specular * light.params.y;
  return vec4<f32>(color, base.a);
}
