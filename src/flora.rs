//! The plants and rocks of the field: what each is shaped like, and where
//! they grow. All of it is made in code - there are no model files.

use crate::app::props::{MeshBuilder, PropInstance, Ribbon, Rng, Shape, facing};
use crate::app::terrain::{self, fbm, height, path_distance};
use glam::{Vec2, Vec3, vec2, vec3};
use std::f32::consts::TAU;

const STEM: Vec3 = Vec3::new(0.16, 0.36, 0.08);
const LEAF: Vec3 = Vec3::new(0.12, 0.38, 0.08);
const ROCK: Vec3 = Vec3::new(0.44, 0.36, 0.31);

/// Everything `populate` decided.
pub struct Scene {
    pub meshes: Vec<MeshBuilder>,
    /// Which mesh, and where.
    pub placements: Vec<(usize, PropInstance)>,
    /// Rocks too big to walk through: centre (x, z) and radius.
    pub obstacles: Vec<(Vec2, f32)>,
    /// Spots where a plant shades out the grass: centre (x, z) and radius.
    pub clearings: Vec<(Vec2, f32)>,
}

/// `count` horizontal directions, evenly spread, starting at `phase`.
fn ring(count: usize, phase: f32) -> impl Iterator<Item = Vec3> {
    (0..count).map(move |i| {
        let angle = phase + TAU * i as f32 / count as f32;
        vec3(angle.cos(), 0.0, angle.sin())
    })
}

/// Leaves rising from the ground around the plant's base.
fn leaves(b: &mut MeshBuilder, rng: &mut Rng, count: usize, length: f32, width: f32, shape: Shape) {
    for out in ring(count, rng.range(0.0, TAU)) {
        b.ribbon(&Ribbon {
            base: out * 0.02,
            out,
            length: length * rng.range(0.7, 1.1),
            width,
            lift: rng.range(0.95, 1.3),
            curl: -rng.range(0.4, 1.0),
            fold: 0.35,
            shape,
            segments: 6,
            ..Default::default()
        });
    }
}

/// A whorl of `count` copies of `petal`, starting `inner` out from the middle.
fn petals(b: &mut MeshBuilder, count: usize, phase: f32, inner: f32, petal: Ribbon) {
    for out in ring(count, phase) {
        b.ribbon(&Ribbon { base: out * inner, out, ..petal });
    }
}

fn tulip(rng: &mut Rng, deep: Vec3, pale: Vec3) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    let top = vec3(rng.range(-0.06, 0.06), rng.range(0.75, 0.95), rng.range(-0.06, 0.06));
    let up = b.stalk(Vec3::ZERO, top, vec3(0.03, 0.0, 0.02), 0.016, STEM);
    b.at(facing(top, up), |b| {
        // Two rings of three: an inner cup, and outer petals just opening.
        for (phase, lift) in [(0.0, 1.0), (TAU / 6.0, 0.85)] {
            let petal = Ribbon {
                length: 0.17,
                width: 0.13,
                lift,
                curl: 0.65,
                fold: 0.6,
                shape: Shape::Petal,
                c0: deep,
                c1: pale,
                rim: Some(pale),
                ..Default::default()
            };
            petals(b, 3, phase, 0.015, petal);
        }
    });
    leaves(&mut b, rng, 3, 0.5, 0.09, Shape::Leaf);
    b.with_sway(1.0)
}

fn daffodil(rng: &mut Rng, petal: Vec3, cup: Vec3) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    let lean = vec3(rng.range(-1.0, 1.0), 0.0, rng.range(-1.0, 1.0)).normalize_or(Vec3::X);
    let top = lean * 0.08 + Vec3::Y * rng.range(0.55, 0.8);
    b.stalk(Vec3::ZERO, top, Vec3::Y * 0.08 - lean * 0.03, 0.013, STEM);
    // The flower nods forward off the top of its stem.
    b.at(facing(top, (lean + Vec3::Y * 0.45).normalize()), |b| {
        let star = Ribbon {
            length: 0.11,
            width: 0.075,
            lift: 0.15,
            curl: -0.25,
            fold: 0.25,
            shape: Shape::Petal,
            c0: petal * 0.9,
            c1: petal,
            segments: 4,
            ..Default::default()
        };
        petals(b, 6, 0.0, 0.01, star);
        b.tube(Vec3::ZERO, Vec3::Y * 0.065, 0.022, 0.04, cup * 0.75, cup, 7);
    });
    leaves(&mut b, rng, 4, 0.5, 0.035, Shape::Sword);
    b.with_sway(1.0)
}

fn iris(rng: &mut Rng, pale: Vec3, deep: Vec3) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    let top = vec3(rng.range(-0.05, 0.05), rng.range(0.85, 1.1), rng.range(-0.05, 0.05));
    let up = b.stalk(Vec3::ZERO, top, vec3(0.02, 0.0, -0.03), 0.017, STEM);
    b.at(facing(top, up), |b| {
        // Three "standards" stand up; three "falls" arch over and hang.
        let standard = Ribbon {
            length: 0.17,
            width: 0.12,
            lift: 1.2,
            curl: 0.5,
            fold: 0.5,
            shape: Shape::Petal,
            c0: pale * 0.85,
            c1: pale,
            ..Default::default()
        };
        petals(b, 3, 0.0, 0.01, standard);
        let fall = Ribbon {
            length: 0.2,
            width: 0.13,
            lift: 0.55,
            curl: -2.1,
            c0: vec3(0.95, 0.8, 0.3),
            c1: deep,
            rim: Some(deep),
            segments: 6,
            ..standard
        };
        petals(b, 3, TAU / 6.0, 0.015, fall);
    });
    // A bud still wrapped, on a short stem of its own.
    let bud = vec3(0.07, top.y * 0.72, 0.03);
    b.stalk(Vec3::Y * top.y * 0.45, bud, Vec3::X * 0.02, 0.01, STEM);
    b.blob(bud + Vec3::Y * 0.04, vec3(0.022, 0.06, 0.022), LEAF, deep);
    leaves(&mut b, rng, 5, 0.8, 0.045, Shape::Sword);
    b.with_sway(1.0)
}

fn snapdragon(rng: &mut Rng, color: Vec3) -> MeshBuilder {
    const FLORETS: usize = 14;
    let mut b = MeshBuilder::default();
    let top = vec3(rng.range(-0.05, 0.05), rng.range(1.0, 1.4), rng.range(-0.05, 0.05));
    b.stalk(Vec3::ZERO, top, Vec3::ZERO, 0.02, STEM);
    // Florets, buds and leaves all spiral up the stem, each a fixed turn on
    // from the last.
    let around = |turn: f32| vec3(turn.cos(), 0.0, turn.sin());

    for i in 0..FLORETS {
        let t = i as f32 / FLORETS as f32;
        let out = around(i as f32 * 2.4);
        let size = 0.06 * (1.0 - 0.55 * t);
        let centre = top * (0.42 + 0.46 * t) + out * size * 0.8;
        b.blob(centre, vec3(size, size * 0.85, size), color * 0.75, color.lerp(Vec3::ONE, 0.25));
        // The lower lip.
        b.ribbon(&Ribbon {
            base: centre + out * size * 0.5 - Vec3::Y * size * 0.3,
            out,
            length: size * 1.5,
            width: size * 1.7,
            lift: -0.1,
            curl: -0.9,
            shape: Shape::Petal,
            c0: color,
            c1: color.lerp(Vec3::ONE, 0.45),
            segments: 3,
            ..Default::default()
        });
    }
    for i in 0..5 {
        let centre = top * (0.9 + 0.02 * i as f32) + around(i as f32 * 2.4 + 1.0) * 0.012;
        b.blob(centre, vec3(0.016, 0.026, 0.016), LEAF, vec3(0.55, 0.75, 0.3));
    }
    for i in 0..9 {
        b.ribbon(&Ribbon {
            base: top * (0.06 + 0.038 * i as f32),
            out: around(i as f32 * 2.4 + 0.7),
            length: rng.range(0.16, 0.24),
            width: 0.055,
            lift: 0.7,
            curl: -0.7,
            segments: 4,
            ..Default::default()
        });
    }
    b.with_sway(1.2)
}

fn daisy(rng: &mut Rng, petal: Vec3, heart: Vec3) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    let stems = 3 + (rng.next() * 3.0) as usize;
    for out in ring(stems, rng.range(0.0, TAU)) {
        let top = out * rng.range(0.08, 0.25) + Vec3::Y * rng.range(0.45, 0.85);
        let up = b.stalk(Vec3::ZERO, top, out * -0.04, 0.012, STEM);
        b.at(facing(top, (up + Vec3::Y * 0.6).normalize()), |b| {
            let ray = Ribbon {
                length: 0.085,
                width: 0.04,
                lift: 0.2,
                curl: -0.35,
                fold: 0.2,
                shape: Shape::Petal,
                c0: petal * 0.8,
                c1: petal,
                segments: 3,
                ..Default::default()
            };
            petals(b, 9, 0.0, 0.02, ray);
            b.blob(Vec3::Y * 0.008, vec3(0.028, 0.014, 0.028), heart * 0.6, heart);
        });
    }
    leaves(&mut b, rng, 5, 0.38, 0.06, Shape::Leaf);
    b.with_sway(1.0)
}

/// One fern frond: an arching rib with a row of leaflets down each side.
fn frond(b: &mut MeshBuilder, out: Vec3, length: f32, lift: f32, curl: f32, dark: Vec3, light: Vec3) {
    const PAIRS: usize = 12;
    let side = out.cross(Vec3::Y).normalize();
    let step = length / PAIRS as f32;
    let mut position = Vec3::ZERO;
    for i in 0..PAIRS {
        let t = i as f32 / PAIRS as f32;
        let angle = lift + curl * t;
        let tangent = out * angle.cos() + Vec3::Y * angle.sin();
        let normal = side.cross(tangent);
        let next = position + tangent * step;
        let color = dark.lerp(light, t);
        b.tube(position, next, 0.009 * (1.0 - t) + 0.002, 0.009 * (1.0 - t), STEM, STEM, 3);

        // Leaflets are longest a third of the way along the frond.
        let reach = length * 0.3 * (1.0 - (t - 0.3).abs() * 1.3).max(0.1);
        for hand in [-1.0, 1.0] {
            let tip = position + side * hand * reach + tangent * reach * 0.35 - normal * reach * 0.15;
            let a = b.push(position, normal, color);
            let c = b.push(next + tangent * step * 0.3, normal, color);
            let d = b.push(tip, normal, color * 1.2);
            b.triangle(a, c, d);
        }
        position = next;
    }
}

fn fern(rng: &mut Rng, dark: Vec3, light: Vec3) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    for out in ring(7, rng.range(0.0, TAU)) {
        let (length, lift, curl) = (rng.range(0.6, 0.95), rng.range(1.0, 1.25), -rng.range(1.3, 1.8));
        frond(&mut b, out, length, lift, curl, dark, light);
    }
    b.with_sway(0.6)
}

/// A low plant of big soft leaves, each on its own stalk.
fn broadleaf(rng: &mut Rng, rim: Option<Vec3>) -> MeshBuilder {
    let mut b = MeshBuilder::default();
    for out in ring(6, rng.range(0.0, TAU)) {
        let stand = out * rng.range(0.1, 0.3) + Vec3::Y * rng.range(0.25, 0.6);
        b.stalk(Vec3::ZERO, stand, out * -0.05, 0.014, STEM);
        b.ribbon(&Ribbon {
            base: stand,
            out,
            length: rng.range(0.35, 0.5),
            width: rng.range(0.26, 0.36),
            lift: 0.3,
            curl: -1.0,
            fold: 0.22,
            c1: LEAF * 1.2,
            rim,
            segments: 7,
            ..Default::default()
        });
    }
    b.with_sway(0.5)
}

/// A faceted rock: a ball with flat cuts. `taper` (0..1) narrows the top; more
/// `detail` means smaller facets.
fn rock(rng: &mut Rng, radii: Vec3, taper: f32, cuts: usize, detail: u32) -> MeshBuilder {
    let planes: Vec<(Vec3, f32)> =
        (0..cuts).map(|_| (rng.direction(), rng.range(0.45, 0.85))).collect();
    // The rock sits with its lower part sunk in the ground.
    let centre = Vec3::Y * radii.y * 0.45;
    let surface = |direction: Vec3| {
        let mut p = direction;
        for &(normal, offset) in &planes {
            let over = p.dot(normal) - offset;
            if over > 0.0 {
                p -= normal * over;
            }
        }
        let narrow = 1.0 - taper * (p.y * 0.5 + 0.5);
        vec3(p.x * narrow, p.y, p.z * narrow) * radii + centre
    };

    // Split each face of an octahedron into small triangles, on the ball.
    fn split(tri: [Vec3; 3], depth: u32, out: &mut Vec<[Vec3; 3]>) {
        if depth == 0 {
            return out.push(tri);
        }
        let [a, b, c] = tri;
        let (ab, bc, ca) = ((a + b).normalize(), (b + c).normalize(), (c + a).normalize());
        for smaller in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
            split(smaller, depth - 1, out);
        }
    }
    let mut triangles = Vec::new();
    for x in [Vec3::X, Vec3::NEG_X] {
        for y in [Vec3::Y, Vec3::NEG_Y] {
            for z in [Vec3::Z, Vec3::NEG_Z] {
                split([x, y, z], detail, &mut triangles);
            }
        }
    }

    let mut b = MeshBuilder::default();
    for tri in triangles {
        let [p0, p1, p2] = tri.map(surface);
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Y);
        if normal.dot(p0 - centre) < 0.0 {
            normal = -normal;
        }
        // Lighter on top, darker underneath, and each facet its own shade -
        // rounded so every triangle of one flat cut gets the same one.
        let facet = (normal * 4.0).round().dot(vec3(12.99, 78.23, 37.72)).sin() * 43_758.547;
        let shade = (0.75 + 0.3 * (normal.y * 0.5 + 0.5)) * (0.84 + 0.3 * facet.fract().abs());
        let corners = [p0, p1, p2].map(|p| b.push(p, normal, ROCK * shade));
        b.triangle(corners[0], corners[1], corners[2]);
    }
    b
}

/// Shapes every plant and rock and decides where each one goes.
pub fn populate() -> Scene {
    let mut rng = Rng::new(7);
    let mut meshes = Vec::new();
    let mut add = |mesh: MeshBuilder| {
        meshes.push(mesh);
        meshes.len() - 1
    };

    let pink = vec3(0.95, 0.3, 0.45);
    let white = vec3(0.95, 0.93, 0.82);
    let orange = vec3(0.95, 0.4, 0.05);
    let yellow = vec3(0.95, 0.75, 0.08);
    let lilac = vec3(0.6, 0.5, 0.95);
    let violet = vec3(0.22, 0.1, 0.6);

    let tulips = [
        add(tulip(&mut rng, pink, vec3(1.0, 0.75, 0.75))),
        add(tulip(&mut rng, vec3(0.9, 0.12, 0.1), orange)),
        add(tulip(&mut rng, yellow, vec3(1.0, 0.9, 0.5))),
    ];
    let daffodils = [
        add(daffodil(&mut rng, white, orange)),
        add(daffodil(&mut rng, yellow, orange)),
        add(daffodil(&mut rng, white, yellow)),
    ];
    let irises = [add(iris(&mut rng, lilac, violet)), add(iris(&mut rng, vec3(0.75, 0.7, 1.0), lilac * 0.6))];
    let snapdragons = [
        add(snapdragon(&mut rng, vec3(0.95, 0.25, 0.55))),
        add(snapdragon(&mut rng, vec3(0.98, 0.55, 0.3))),
        add(snapdragon(&mut rng, white)),
        add(snapdragon(&mut rng, vec3(0.6, 0.2, 0.7))),
    ];
    let daisies = [
        add(daisy(&mut rng, orange, vec3(0.5, 0.2, 0.02))),
        add(daisy(&mut rng, vec3(0.85, 0.06, 0.12), yellow)),
        add(daisy(&mut rng, vec3(0.2, 0.3, 0.95), vec3(0.1, 0.1, 0.4))),
        add(daisy(&mut rng, yellow, orange)),
        add(daisy(&mut rng, white, yellow)),
    ];
    let foliage = [
        add(fern(&mut rng, vec3(0.02, 0.2, 0.06), vec3(0.1, 0.45, 0.1))),
        add(fern(&mut rng, vec3(0.2, 0.4, 0.03), vec3(0.6, 0.75, 0.08))),
        add(broadleaf(&mut rng, None)),
        add(broadleaf(&mut rng, Some(vec3(0.8, 0.4, 0.05)))),
    ];
    let boulders = [
        add(rock(&mut rng, vec3(0.8, 1.5, 0.8), 0.55, 22, 3)), // spire
        add(rock(&mut rng, vec3(1.0, 0.8, 0.9), 0.1, 26, 3)),  // block
        add(rock(&mut rng, vec3(1.3, 0.5, 0.9), 0.2, 22, 3)),  // slab
        add(rock(&mut rng, vec3(0.9, 0.9, 1.1), 0.25, 28, 3)),
    ];
    let pebbles = [
        add(rock(&mut rng, vec3(1.0, 0.6, 0.8), 0.2, 8, 2)),
        add(rock(&mut rng, vec3(0.8, 0.5, 1.0), 0.0, 6, 2)),
    ];
    let flowers: [&[usize]; 5] = [&tulips, &daffodils, &irises, &snapdragons, &daisies];

    let edge = terrain::SIZE / 2.0 - 3.0;
    let mut placements = Vec::new();
    let mut obstacles: Vec<(Vec2, f32)> = Vec::new();
    let mut clearings = Vec::new();
    // `sink` is how deep it is set into the ground, as a share of its scale.
    let mut place = |rng: &mut Rng, mesh: usize, at: Vec2, scale: f32, sink: f32| {
        let shade = rng.range(0.88, 1.08);
        placements.push((
            mesh,
            PropInstance {
                position: [at.x, height(at.x, at.y) - sink * scale, at.y],
                scale,
                yaw: rng.range(0.0, TAU),
                tint: [shade; 3],
            },
        ));
    };
    let offset = |rng: &mut Rng, distance: f32| Vec2::from_angle(rng.range(0.0, TAU)) * distance;

    // Rocks: outcrops of one big rock, a few lesser ones and some rubble.
    // Kept off the paths and away from where the player starts.
    for _ in 0..55 {
        let centre = vec2(rng.range(-edge, edge), rng.range(-edge, edge));
        if path_distance(centre.x, centre.y) < 4.5 || centre.length() < 7.0 {
            continue;
        }
        let size = rng.range(0.8, 1.7);
        let mesh = rng.pick(&boulders);
        place(&mut rng, mesh, centre, size, 0.12);
        obstacles.push((centre, size * 0.85));
        for _ in 0..(rng.next() * 3.5) as usize {
            let (reach, scale, mesh) = (size * rng.range(1.1, 1.7), size * 0.5, rng.pick(&boulders));
            let at = centre + offset(&mut rng, reach);
            place(&mut rng, mesh, at, scale, 0.12);
            obstacles.push((at, scale * 0.85));
        }
        for _ in 0..3 + (rng.next() * 5.0) as usize {
            let (reach, scale, mesh) = (size * rng.range(1.0, 2.6), rng.range(0.1, 0.28), rng.pick(&pebbles));
            let at = centre + offset(&mut rng, reach);
            place(&mut rng, mesh, at, scale, 0.1);
        }
    }
    // Loose stones scattered along the paths.
    for _ in 0..500 {
        let at = vec2(rng.range(-edge, edge), rng.range(-edge, edge));
        if path_distance(at.x, at.y) < 2.2 {
            let (scale, mesh) = (rng.range(0.05, 0.16), rng.pick(&pebbles));
            place(&mut rng, mesh, at, scale, 0.1);
        }
    }

    // Plants: one try per cell of a loose grid. Each kind of flower grows
    // in drifts of its own; ferns and big leaves gather round the rocks.
    let spacing = 1.1;
    let cells = (2.0 * edge / spacing) as i32;
    for j in 0..cells {
        for i in 0..cells {
            let at = vec2(i as f32 + rng.next(), j as f32 + rng.next()) * spacing - Vec2::splat(edge);
            if path_distance(at.x, at.y) < 1.7 {
                continue;
            }
            let to_rock = obstacles.iter().map(|(c, r)| c.distance(at) - r).fold(f32::MAX, f32::min);
            if to_rock < 0.25 {
                continue;
            }

            let drift = |kind: usize| fbm(at * 0.03 + Vec2::splat(kind as f32 * 37.0));
            let kind = (0..flowers.len()).max_by(|&a, &b| drift(a).total_cmp(&drift(b))).unwrap();
            let roll = rng.next();
            let leafy = (to_rock < 1.8 && roll < 0.3) || roll < 0.006;
            let mesh = if leafy {
                rng.pick(&foliage)
            } else if drift(kind) > 0.6 && roll < 0.5 {
                rng.pick(flowers[kind])
            } else if roll < 0.016 {
                rng.pick(flowers[(roll * 1e4) as usize % flowers.len()])
            } else {
                continue;
            };
            let scale = rng.range(0.8, 1.3);
            place(&mut rng, mesh, at, scale, 0.0);
            if leafy {
                clearings.push((at, 0.3 * scale));
            }
        }
    }

    Scene { meshes, placements, obstacles, clearings }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_field_is_populated_sensibly() {
        let scene = populate();
        assert!(scene.placements.len() > 2000, "only {} props", scene.placements.len());
        assert!(scene.obstacles.len() > 20);

        for mesh in &scene.meshes {
            assert!(!mesh.indices.is_empty());
            assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertices.len()));
            assert!(mesh.vertices.iter().all(|v| {
                v.position.iter().chain(&v.normal).chain(&v.color).all(|f| f.is_finite())
            }));
        }
        for (mesh, instance) in &scene.placements {
            assert!(*mesh < scene.meshes.len());
            assert!(terrain::contains(instance.position[0], instance.position[2]));
        }
        // Big rocks stay off the paths and clear of the start.
        for (centre, radius) in &scene.obstacles {
            assert!(path_distance(centre.x, centre.y) > *radius);
            assert!(centre.length() > 3.0);
        }
    }
}
