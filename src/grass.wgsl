// Procedural grass, after Sucker Punch's "Procedural Grass in Ghost of
// Tsushima" (GDC 2021).
//
//   cs_main  every frame, decides where blades are and what each looks like,
//            and appends the visible ones to a buffer
//   vs_main  turns one blade's parameters into a curved strip of triangles
//   fs_main  shades it
//
// There is no grass mesh anywhere: a blade is 64 bytes of parameters.

// One kind of grass, as multipliers on the settings every kind shares.
struct GrassType {
  a: vec4<f32>,  // height, width, tilt, bend
  b: vec4<f32>,  // x: density, y: wind response, z: how far it takes on its own colour, w: ear size
  c: vec4<f32>,  // rgb: its own colour, a: how much of the field it covers
}

const SHORT: u32 = 0u;
const REGULAR: u32 = 1u;
const TALL: u32 = 2u;
const GRAIN: u32 = 3u;

struct Grass {
  view_proj: mat4x4<f32>,
  eye: vec4<f32>,    // xyz: camera position, w: time in seconds
  wind: vec4<f32>,   // x, y: direction on the ground (xz), z: strength, w: speed
  lod: vec4<f32>,    // x: high-detail radius, y: far radius, z: blade spacing, w: field size
  shape: vec4<f32>,  // x: height, y: width, z: tilt, w: bend
  clump: vec4<f32>,  // x: clump size, y: height variation, z: facing pull, w: colour variation
  root: vec4<f32>,   // rgb: colour at the ground, a: ambient occlusion at the ground
  tip: vec4<f32>,    // rgb: colour at the tip, a: translucency
  misc: vec4<f32>,   // x: specular, y: number of pushers
  dry: vec4<f32>,    // rgb: dried-out colour, a: how much of the field is dry
  flower: vec4<f32>, // rgb: first flower colour, a: how many flowers
  flower2: vec4<f32>,// rgb: second flower colour, a: flower head size
  vary: vec4<f32>,   // x: patch height variation, y: patch thinning, z: patch size, w: density
  ground: vec4<f32>, // x: path width, y: how much of the field is worn to dirt
  soil: vec4<f32>,   // rgb: colour of bare dirt
  vary2: vec4<f32>,  // x: stragglers, y: tussocks, z: swell
  types: array<GrassType, 4>,
  pushers: array<vec4<f32>, 8>, // xyz: position, w: radius
}

struct Blade {
  position: vec3<f32>,
  facing: f32,    // angle around Y the blade leans towards
  height: f32,
  width: f32,
  tilt: f32,      // how far the tip leans over, in radians from vertical
  bend: f32,      // how much the middle bows out
  hash: f32,      // 0..1, unique to the blade
  clump: f32,     // 0..1, shared by the blade's clump
  wind: f32,      // wind strength where the blade stands
  segments: f32,  // 7 near the camera, 3 far away
  kind: f32,      // 0: grass, 1 and 2: a flower of the first or second colour, 3: grain
  species: f32,   // which GrassType
  dry: f32,       // 0: green, 1: dried out
}

// Laid out as indirect draw arguments, so the draw call reads the blade
// count straight from the GPU. The CPU never learns it.
struct DrawArgs {
  vertex_count: u32,
  instance_count: atomic<u32>,
  first_vertex: u32,
  first_instance: u32,
}

const TAU: f32 = 6.2831853;
const UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

// ---------------------------------------------------------------- shared

// Must match terrain::height in terrain.rs, or the grass floats.
fn terrain_height(p: vec2<f32>) -> f32 {
  return 3.0 * sin(p.x * 0.045 + 0.5) * cos(p.y * 0.038 - 1.0)
    + 1.5 * sin(p.x * 0.09 + p.y * 0.07 + 2.0)
    + 0.6 * sin(p.x * 0.21 - p.y * 0.17);
}

// Metres to the middle of the nearest dirt path. Must match
// terrain::path_distance in terrain.rs.
fn path_distance(p: vec2<f32>) -> f32 {
  let a = p.y - (14.0 * sin(p.x * 0.031 + 0.4) + 6.0 * sin(p.x * 0.083 + 2.0) - 9.0);
  let a_slope = 14.0 * 0.031 * cos(p.x * 0.031 + 0.4) + 6.0 * 0.083 * cos(p.x * 0.083 + 2.0);
  let b = p.x - (11.0 * sin(p.y * 0.037 + 1.3) + 5.0 * sin(p.y * 0.097 + 0.2) - 24.0);
  let b_slope = 11.0 * 0.037 * cos(p.y * 0.037 + 1.3) + 5.0 * 0.097 * cos(p.y * 0.097 + 0.2);
  return min(abs(a) / sqrt(1.0 + a_slope * a_slope), abs(b) / sqrt(1.0 + b_slope * b_slope));
}

// Integer hash (PCG): the same cell always gives the same blade.
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash_cell(cell: vec2<i32>, salt: u32) -> u32 {
  return pcg(pcg(bitcast<u32>(cell.x) + salt) + bitcast<u32>(cell.y));
}

// Four independent values in 0..1 from one seed.
fn rand4(seed: u32) -> vec4<f32> {
  let a = pcg(seed);
  let b = pcg(a);
  let c = pcg(b);
  let d = pcg(c);
  return vec4<f32>(vec4<u32>(a, b, c, d) >> vec4<u32>(8u)) / 16777216.0;
}

// Smooth noise in 0..1, for wind gusts.
fn noise(p: vec2<f32>) -> f32 {
  let cell = vec2<i32>(floor(p));
  let f = fract(p);
  let s = f * f * (3.0 - 2.0 * f);
  let a = rand4(hash_cell(cell, 7u)).x;
  let b = rand4(hash_cell(cell + vec2<i32>(1, 0), 7u)).x;
  let c = rand4(hash_cell(cell + vec2<i32>(0, 1), 7u)).x;
  let d = rand4(hash_cell(cell + vec2<i32>(1, 1), 7u)).x;
  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// Three layers of noise: big soft shapes with ragged edges.
fn fbm(p: vec2<f32>) -> f32 {
  return noise(p) * 0.6 + noise(p * 2.3 + 17.0) * 0.3 + noise(p * 5.1 - 9.0) * 0.1;
}

// 0 outside a patch, 1 inside. `amount` (0..1) is how much of the field the
// patches cover.
fn patches(p: vec2<f32>, amount: f32) -> f32 {
  let threshold = mix(0.85, 0.25, amount);
  return smoothstep(threshold - 0.08, threshold + 0.08, fbm(p));
}

// 0: grass, 1: dirt (paths, worn patches, cleared areas). `density` is the painted map.
fn dirt(p: vec2<f32>, density: f32, path_width: f32, worn: f32) -> f32 {
  let ragged = (fbm(p * 0.8) - 0.5) * 1.6;
  let half_width = path_width * 0.5;
  let path = (1.0 - smoothstep(half_width - 0.2, half_width + 0.5, path_distance(p) + ragged))
    * step(0.01, path_width);
  let worn_through = patches(p * 0.05 + 91.0, worn) * 0.9;
  return max(max(path, worn_through), 1.0 - density);
}

// 1 beside the paths and round worn dirt, fading to 0 a few metres out:
// where feet keep the grass short.
fn trodden(p: vec2<f32>, path_width: f32, worn: f32) -> f32 {
  let ragged = (fbm(p * 0.8) - 0.5) * 1.6;
  let half_width = path_width * 0.5;
  let beside = (1.0 - smoothstep(half_width + 0.4, half_width + 2.6, path_distance(p) + ragged))
    * step(0.01, path_width);
  return max(beside, patches(p * 0.05 + 91.0, min(worn + 0.12, 1.0)));
}

// --------------------------------------------------------------- compute

@group(0) @binding(0) var<uniform> gen: Grass;
@group(0) @binding(1) var<storage, read_write> near_blades: array<Blade>;
@group(0) @binding(2) var<storage, read_write> far_blades: array<Blade>;
@group(0) @binding(3) var<storage, read_write> draws: array<DrawArgs, 2>;
@group(0) @binding(4) var density_map: texture_2d<f32>;
@group(0) @binding(5) var density_sampler: sampler;

// One thread per possible blade: id.xy is a grid cell under the camera, id.z the
// LOD (0 near, every cell; 1 far, every second cell).
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let lod = id.z;
  let spacing = gen.lod.z;
  let stride = i32(1u << lod);
  let radius = select(gen.lod.x, gen.lod.y, lod == 1u);
  let side = u32(ceil(2.0 * radius / (spacing * f32(stride))));
  if id.x >= side || id.y >= side {
    return;
  }

  // Cell coordinates on the fine grid, in world space - so a blade stays put
  // as the camera, and the grid with it, moves.
  let centre = vec2<i32>(floor(gen.eye.xz / (spacing * f32(stride))));
  let fine = (centre + vec2<i32>(id.xy) - vec2<i32>(i32(side / 2u))) * stride;

  // Cells with two even coordinates exist at both levels of detail, with the
  // same seed, so a blade doesn't jump when it changes level.
  let coarse = (fine.x & 1) == 0 && (fine.y & 1) == 0;
  let seed = hash_cell(fine, 0u);
  let r = rand4(seed);
  let r2 = rand4(seed ^ 0x9e3779b9u);
  let r3 = rand4(seed ^ 0x7f4a7c15u);

  let jitter = select(1.0, 2.0, coarse);
  let ground = (vec2<f32>(fine) + r.xy * jitter) * spacing;
  let field = gen.lod.w * 0.5;
  if abs(ground.x) > field || abs(ground.y) > field {
    return;
  }
  var position = vec3<f32>(ground.x, terrain_height(ground), ground.y);

  // Cheap, generous frustum cull before the costly work; exact test comes later.
  let early = gen.view_proj * vec4<f32>(position + UP, 1.0);
  if abs(early.x) > early.w + 6.0 || abs(early.y) > early.w + 6.0 || early.w < -6.0 {
    return;
  }

  // Distance decides the level of detail. Approaching the switch, three
  // blades in four shrink away and the survivor widens to cover for them.
  let range = distance(position, gen.eye.xyz);
  let near_limit = gen.lod.x;
  if (lod == 0u && range >= near_limit) || (lod == 1u && range < near_limit) {
    return;
  }
  let thinning = smoothstep(0.6 * near_limit, near_limit, range);
  if lod == 0u && !coarse && thinning >= 0.99 {
    return;
  }
  var height_scale = 1.0 - smoothstep(0.8 * gen.lod.y, gen.lod.y, range);
  var width_scale = 1.0;
  if coarse {
    width_scale = 1.0 + thinning;
  } else {
    height_scale *= 1.0 - thinning;
  }

  // Painted density: 0 is bare ground.
  let density = textureSampleLevel(density_map, density_sampler, ground / gen.lod.w + 0.5, 0.0).r;
  height_scale *= density;
  // Grass gets shorter towards bare ground, then stops.
  height_scale *= 1.0 - smoothstep(0.1, 0.55, dirt(ground, 1.0, gen.ground.x, gen.ground.y));

  // Patches, tens of metres across: the field grows tall and thick in some
  // places, short and sparse in others, and dries out in others again.
  let patch_pos = ground / gen.vary.z;
  let lush = smoothstep(0.3, 0.7, fbm(patch_pos));
  let dryness = patches(patch_pos * 0.7 + 31.7, gen.dry.a);
  height_scale *= 1.0 + (lush - 0.5) * 2.0 * gen.vary.x;
  height_scale *= 1.0 - 0.3 * dryness;
  if r3.x < gen.vary.y * (1.0 - lush) || r3.w > gen.vary.w {
    return;
  }
  if height_scale < 0.05 {
    return;
  }

  // Which kind grows here: short where trodden, grain in dry patches, tall where
  // lush. Each blade rolls for itself, so kinds mingle at borders.
  let r4 = rand4(seed ^ 0x2545f491u);
  let r5 = rand4(seed ^ 0x68e31da4u);
  let short_cover = gen.types[SHORT].c.w;
  var species = REGULAR;
  if r4.x < trodden(ground, gen.ground.x, gen.ground.y) * min(short_cover * 8.0, 1.0)
      || r4.x < patches(patch_pos * 1.4 + 113.0, short_cover * 0.6) {
    species = SHORT;
  } else if r4.y < patches(patch_pos * 0.7 + 31.7, gen.types[GRAIN].c.w) {
    species = GRAIN;
  } else if r4.z < patches(patch_pos, gen.types[TALL].c.w) {
    species = TALL;
  }
  let kind_of = gen.types[species];
  if r4.w > kind_of.b.x {
    return;
  }

  // Clumps: one random point per cell, nearest wins (Voronoi). Blades of a clump
  // share height and colour and splay outward, breaking the lawn look.
  let clump_pos = ground / gen.clump.x;
  let clump_cell = vec2<i32>(floor(clump_pos));
  var nearest = 1e9;
  var clump_seed = 0u;
  var clump_centre = vec2<f32>(0.0);
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let cell = clump_cell + vec2<i32>(x, y);
      let cell_seed = hash_cell(cell, 1u);
      let point = vec2<f32>(cell) + rand4(cell_seed).xy;
      let d = distance(point, clump_pos);
      if d < nearest {
        nearest = d;
        clump_seed = cell_seed;
        clump_centre = point * gen.clump.x;
      }
    }
  }
  let c = rand4(clump_seed ^ 0x51ed270bu);
  let c2 = rand4(clump_seed ^ 0x3c6ef372u);

  let angle = r.z * TAU;
  var facing = vec2<f32>(cos(angle), sin(angle));
  let outward = ground - clump_centre;
  if length(outward) > 0.001 {
    facing = normalize(mix(facing, normalize(outward), gen.clump.z * c.z));
  }

  var blade: Blade;
  var tallness = (0.7 + 0.6 * r.w) * (1.0 + (c.x - 0.5) * 2.0 * gen.clump.y);
  // Most blades are much of a height; a few stragglers stand well clear.
  tallness *= 1.0 + gen.vary2.x * 1.5 * pow(r5.x, 8.0);
  // Now and then a whole clump is a tussock: taller, with broader blades.
  let tussock = step(1.0 - 0.3 * gen.vary2.y, c2.x);
  tallness *= 1.0 + tussock * (0.5 + 0.5 * c2.y);
  // And the field swells and dips every few metres.
  tallness *= 1.0 + (noise(ground / 3.0) - 0.5) * 2.0 * gen.vary2.z;
  // The variations stack; this stops the unluckiest blade becoming a tree.
  tallness = min(tallness, 2.0);

  blade.height = gen.shape.x * kind_of.a.x * tallness * height_scale;
  blade.width = gen.shape.y * kind_of.a.y * (0.8 + 0.4 * r2.x) * width_scale * (1.0 + 0.3 * tussock);
  blade.tilt = gen.shape.z * kind_of.a.z * (0.5 + r2.y);
  blade.bend = gen.shape.w * kind_of.a.w * (0.5 + r2.z);
  blade.species = f32(species);
  blade.hash = r2.w;
  blade.clump = c.y;
  blade.segments = select(7.0, 3.0, lod == 1u);
  blade.dry = dryness * (0.55 + 0.45 * c.w);

  // Flowers grow in drifts, with a few strays everywhere else. A flower is
  // a blade that stands taller and straighter and ends in a head.
  let drift = patches(patch_pos * 1.9 + 71.3, 0.45);
  let chance = gen.flower.a * 0.1 * (0.04 + 0.96 * drift);
  blade.kind = 0.0;
  if species == GRAIN {
    blade.kind = 3.0;
    blade.dry = 0.0;
  }
  if r3.y < chance && species == REGULAR {
    blade.kind = select(1.0, 2.0, fbm(patch_pos * 1.3 - 53.1) > 0.5);
    blade.height *= 1.2 + 0.2 * r3.z;
    blade.tilt *= 0.5;
    blade.bend *= 0.5;
    blade.dry = 0.0;
  }

  // Anything standing in the grass flattens it away from itself.
  for (var i = 0; i < i32(gen.misc.y); i++) {
    let pusher = gen.pushers[i];
    let away = position - pusher.xyz;
    let reach = 1.0 - smoothstep(0.3 * pusher.w, pusher.w, length(away));
    if reach > 0.0 && length(away.xz) > 0.001 {
      facing = normalize(mix(facing, normalize(away.xz), reach));
      blade.tilt = mix(blade.tilt, 1.35, reach);
    }
  }
  blade.facing = atan2(facing.y, facing.x);

  // Wind: gusts are a noise pattern sliding across the field.
  let blown = gen.wind.xy * gen.eye.w * gen.wind.w;
  let gust = noise(ground * 0.07 - blown * 0.35) * 0.7 + noise(ground * 0.3 - blown) * 0.3;
  blade.wind = gen.wind.z * (0.25 + 0.75 * gust) * kind_of.b.y;
  blade.position = position;

  // Skip blades the camera can't see. The margin keeps a blade whose root is
  // just off screen but whose tip isn't.
  let clip = gen.view_proj * vec4<f32>(position + UP * blade.height * 0.5, 1.0);
  let margin = 2.0 + blade.height;
  if abs(clip.x) > clip.w + margin || abs(clip.y) > clip.w + margin || clip.w < -margin {
    return;
  }

  let slot = atomicAdd(&draws[lod].instance_count, 1u);
  if lod == 0u {
    if slot < arrayLength(&near_blades) {
      near_blades[slot] = blade;
    }
  } else if slot < arrayLength(&far_blades) {
    far_blades[slot] = blade;
  }
}

// ---------------------------------------------------------------- render

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

@group(1) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(0) var<uniform> light: Light;
@group(2) @binding(1) var t_shadow: texture_depth_2d;
@group(2) @binding(2) var s_shadow: sampler_comparison;
@group(3) @binding(0) var<uniform> grass: Grass;
@group(3) @binding(1) var<storage, read> blades: array<Blade>;
@group(3) @binding(2) var ground_density: texture_2d<f32>;
@group(3) @binding(3) var ground_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  // x: 0 at the root, 1 at the tip. y: -1 and 1 at the edges.
  @location(2) along: vec2<f32>,
  // x: per-blade random, y: per-clump random
  @location(3) variation: vec2<f32>,
  // x: dryness, y: kind (see Blade), z: 1 on a flower's head
  @location(4) extra: vec3<f32>,
  // rgb: the colour of this kind of grass, a: how far the blade takes it on
  @location(5) tone: vec4<f32>,
}

fn bezier(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

fn bezier_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let u = 1.0 - t;
  return 3.0 * u * u * (p1 - p0) + 6.0 * u * t * (p2 - p1) + 3.0 * t * t * (p3 - p2);
}

// A blade is a triangle strip: pairs of vertices up its two edges, then one
// at the tip. Its spine is a cubic Bezier curve from root to tip.
@vertex
fn vs_main(
  @builtin(vertex_index) vertex: u32,
  @builtin(instance_index) instance: u32,
) -> VertexOutput {
  var out: VertexOutput;
  // More blades were wanted than the buffer holds: draw nothing for the rest.
  if instance >= arrayLength(&blades) {
    out.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let blade = blades[instance];

  let segments = u32(blade.segments);
  let ring = min(vertex >> 1u, segments);
  let t = f32(ring) / f32(segments);
  let edge = select(f32(vertex & 1u) * 2.0 - 1.0, 0.0, ring == segments);

  let facing = vec3<f32>(cos(blade.facing), 0.0, sin(blade.facing));
  let side = cross(UP, facing);

  // Where the tip ends up: leaning over by `tilt`, then shoved by the wind
  // and bobbing in it, then pulled back to the blade's length.
  let time = grass.eye.w;
  let wind_dir = vec3<f32>(grass.wind.x, 0.0, grass.wind.y);
  let sway = sin(time * (2.0 + 2.0 * grass.wind.w) + blade.hash * TAU) * 0.12 * blade.wind;
  var reach = facing * sin(blade.tilt) + UP * cos(blade.tilt);
  reach += wind_dir * (blade.wind * 0.6 + sway) + side * sway * 0.5;
  reach = normalize(reach);
  reach.y = max(reach.y, 0.05);

  // The two middle control points sit on the root-tip line, pushed out
  // sideways from it by `bend` to bow the blade.
  let bow = normalize(cross(reach, side)) * blade.bend * blade.height;
  let p0 = blade.position;
  let p3 = p0 + reach * blade.height;
  let p1 = mix(p0, p3, 0.33) + bow;
  let p2 = mix(p0, p3, 0.66) + bow;

  var position = bezier(p0, p1, p2, p3, t);
  let tangent = normalize(bezier_tangent(p0, p1, p2, p3, t));
  var normal = normalize(cross(side, tangent));

  // Width: full at the root, pinching in towards the tip.
  var half_width = blade.width * 0.5 * (1.0 - t * t);
  var head = 0.0;
  if blade.kind > 2.5 {
    // Grain: a thin stalk carrying an ear over its top rings.
    let ear = ring + select(1u, 2u, segments > 3u) >= segments;
    half_width = select(blade.width * 0.2, grass.types[GRAIN].b.w * 0.5, ear);
    head = f32(ear);
  } else if blade.kind > 0.5 {
    // A flower: a thin stem that spreads into a head at the last ring.
    half_width = select(blade.width * 0.15, grass.flower2.a * 0.5, ring + 1u == segments);
    head = f32(ring + 1u >= segments);
  }
  position += side * edge * half_width;

  // A blade seen edge-on would vanish to a line. Widen it across the screen
  // instead, so a field never thins out as the camera turns.
  let to_eye = normalize(camera.eye.xyz - position);
  let edge_on = 1.0 - abs(dot(normal, to_eye));
  let screen_right = normalize(cross(to_eye, tangent));
  position += screen_right * edge * half_width * edge_on * edge_on * sign(dot(screen_right, side));

  // Tilt the normal outwards at the edges, as if the blade were rounded.
  normal = normalize(normal + side * edge * 0.45);

  out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
  out.world_position = position;
  out.normal = normal;
  out.along = vec2<f32>(t, edge);
  out.variation = vec2<f32>(blade.hash, blade.clump);
  out.extra = vec3<f32>(blade.dry, blade.kind, head);
  let kind_of = grass.types[u32(blade.species)];
  out.tone = vec4<f32>(kind_of.c.rgb, kind_of.b.z);
  return out;
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
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
  let t = in.along.x;
  let to_eye = camera.eye.xyz - in.world_position;
  let view = normalize(to_eye);
  let to_light = normalize(-light.direction.xyz);

  // Both faces are drawn; light the one being looked at.
  var normal = normalize(in.normal);
  if !front {
    normal = -normal;
  }
  // Far away, thousands of tiny blades each glinting differently just
  // sparkle. Lean them all towards one shared normal instead.
  let far = smoothstep(0.3 * grass.lod.x, grass.lod.y, length(to_eye));
  normal = normalize(mix(normal, UP, 0.75 * far));

  // Colour: root to tip, shifted per clump and a little per blade, with a
  // darker vein down the middle.
  var base = mix(grass.root.rgb, grass.tip.rgb, t * t);
  base = mix(base, grass.dry.rgb * (0.35 + 0.65 * t), in.extra.x);
  base = mix(base, in.tone.rgb * (0.35 + 0.65 * t), in.tone.a);
  base *= 1.0 + (in.variation.y - 0.5) * 2.0 * grass.clump.w;
  // Some clumps lean yellow, some blue.
  let hue = mix(vec3<f32>(1.12, 1.0, 0.75), vec3<f32>(0.82, 1.0, 1.15), fract(in.variation.y * 7.31));
  base *= mix(vec3<f32>(1.0), hue, min(grass.clump.w * 2.5, 1.0));
  base *= 0.9 + 0.2 * in.variation.x;
  base *= 1.0 - 0.18 * (1.0 - abs(in.along.y)) * (1.0 - far);
  if in.extra.z > 0.5 {
    base = select(grass.flower.rgb, grass.flower2.rgb, in.extra.y > 1.5);
    if in.extra.y > 2.5 {
      base = in.tone.rgb * 1.15;
    }
  }

  // Roots sit in each other's shade.
  let occlusion = mix(grass.root.a, 1.0, t);
  let lit = shadow(in.world_position);

  let facing_light = dot(normal, to_light);
  let diffuse = max(facing_light, 0.0);
  // Light coming through the blade from behind.
  let through = max(-facing_light, 0.0) * grass.tip.a;
  let halfway = normalize(to_light + view);
  let specular = pow(max(dot(normal, halfway), 0.0), 48.0) * grass.misc.x * t * (1.0 - far);

  let ambient = light.params.x;
  let direct = (diffuse + through) * lit;
  let color = base * light.color.rgb * (ambient + direct) * occlusion
    + light.color.rgb * specular * lit * step(0.0001, diffuse);
  return vec4<f32>(color, 1.0);
}

// ---------------------------------------------------------------- ground

// The ground is drawn here too: it uses the same functions that place the grass.

struct GroundOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
}

const GROUND_CELLS: u32 = 200u;

// No mesh: vertex n is a corner of square n / 6 on a grid across the field.
@vertex
fn vs_ground(@builtin(vertex_index) vertex: u32) -> GroundOutput {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
  );
  let square = vertex / 6u;
  let grid = vec2<f32>(f32(square % GROUND_CELLS), f32(square / GROUND_CELLS)) + corners[vertex % 6u];
  let p = (grid / f32(GROUND_CELLS) - 0.5) * grass.lod.w;

  var out: GroundOutput;
  out.world_position = vec3<f32>(p.x, terrain_height(p), p.y);
  out.clip_position = camera.view_proj * vec4<f32>(out.world_position, 1.0);
  return out;
}

@fragment
fn fs_ground(in: GroundOutput) -> @location(0) vec4<f32> {
  let p = in.world_position.xz;
  let e = 0.25;
  let normal = normalize(vec3<f32>(
    terrain_height(p - vec2<f32>(e, 0.0)) - terrain_height(p + vec2<f32>(e, 0.0)),
    2.0 * e,
    terrain_height(p - vec2<f32>(0.0, e)) - terrain_height(p + vec2<f32>(0.0, e)),
  ));
  // Fine detail would shimmer in the distance, so it fades out.
  let detail = 1.0 - smoothstep(25.0, 70.0, distance(in.world_position, camera.eye.xyz));

  let density = textureSample(ground_density, ground_sampler, p / grass.lod.w + 0.5).r;
  let bare = dirt(p, density, grass.ground.x, grass.ground.y);

  // Turf: what the grass looks like from too far away to see blades,
  // drying out in the same patches the blades do.
  var turf = mix(grass.root.rgb, grass.tip.rgb, 0.3);
  let dryness = patches(p / grass.vary.z * 0.7 + 31.7, grass.dry.a);
  turf = mix(turf, grass.dry.rgb * 0.6, dryness * 0.8);
  turf *= mix(1.0, 0.85 + 0.3 * noise(p * 2.5), detail);

  // Dirt: mottled in broad stains, grainy up close, with small pale stones.
  var soil = grass.soil.rgb * (0.7 + 0.6 * fbm(p * 0.5));
  soil *= mix(1.0, 0.85 + 0.3 * noise(p * 7.0), detail);
  let stones = smoothstep(0.8, 0.86, noise(p * 19.0)) * detail;
  soil = mix(soil, soil * 1.6 + 0.03, stones * 0.7);

  var color = mix(turf, soil, smoothstep(0.3, 0.7, bare));
  // Dark, damp earth where the grass gives out.
  color *= 1.0 - 0.3 * smoothstep(0.2, 0.5, bare) * (1.0 - smoothstep(0.5, 0.9, bare));

  let to_light = normalize(-light.direction.xyz);
  let diffuse = max(dot(normal, to_light), 0.0) * shadow(in.world_position);
  return vec4<f32>(color * light.color.rgb * (light.params.x + diffuse), 1.0);
}
