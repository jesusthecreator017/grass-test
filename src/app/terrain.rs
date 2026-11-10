//! A rolling field as a plain function of (x, z), so ground, props and picking agree.
// ponytail: analytic; swap for heightmap and mask lookups to author the field.

use glam::{Vec2, vec2};

/// Side length of the square field, centred on the origin.
pub const SIZE: f32 = 200.0;

// grass.wgsl has its own copy of this (terrain_height); a test keeps them equal.
pub fn height(x: f32, z: f32) -> f32 {
    3.0 * (x * 0.045 + 0.5).sin() * (z * 0.038 - 1.0).cos()
        + 1.5 * (x * 0.09 + z * 0.07 + 2.0).sin()
        + 0.6 * (x * 0.21 - z * 0.17).sin()
}

/// Metres to the middle of the nearest dirt path. Two paths wind across the
/// field, one each way. grass.wgsl has its own copy of this too.
pub fn path_distance(x: f32, z: f32) -> f32 {
    let a = z - (14.0 * (x * 0.031 + 0.4).sin() + 6.0 * (x * 0.083 + 2.0).sin() - 9.0);
    let a_slope = 14.0 * 0.031 * (x * 0.031 + 0.4).cos() + 6.0 * 0.083 * (x * 0.083 + 2.0).cos();
    let b = x - (11.0 * (z * 0.037 + 1.3).sin() + 5.0 * (z * 0.097 + 0.2).sin() - 24.0);
    let b_slope = 11.0 * 0.037 * (z * 0.037 + 1.3).cos() + 5.0 * 0.097 * (z * 0.097 + 0.2).cos();
    (a.abs() / (1.0 + a_slope * a_slope).sqrt()).min(b.abs() / (1.0 + b_slope * b_slope).sqrt())
}

pub fn contains(x: f32, z: f32) -> bool {
    x.abs() <= SIZE / 2.0 && z.abs() <= SIZE / 2.0
}

fn hash(x: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (z as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h >> 8) as f32 / 16_777_216.0
}

/// Smooth noise in 0..1. For deciding where things grow; it is not the
/// noise the shaders use.
pub fn noise(p: Vec2) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let s = f * f * (3.0 - 2.0 * f);
    let (x, z) = (cell.x as i32, cell.y as i32);
    let top = hash(x, z) + (hash(x + 1, z) - hash(x, z)) * s.x;
    let bottom = hash(x, z + 1) + (hash(x + 1, z + 1) - hash(x, z + 1)) * s.x;
    top + (bottom - top) * s.y
}

/// Three layers of `noise`: big soft shapes with ragged edges.
pub fn fbm(p: Vec2) -> f32 {
    noise(p) * 0.6 + noise(p * 2.3 + vec2(17.0, 17.0)) * 0.3 + noise(p * 5.1 - vec2(9.0, 9.0)) * 0.1
}

/// Where a ray first goes under the ground, if it does so inside the field.
pub fn raycast(origin: glam::Vec3, direction: glam::Vec3) -> Option<glam::Vec3> {
    let below = |t: f32| {
        let p = origin + direction * t;
        p.y < height(p.x, p.z)
    };

    // March until the ray dips under the surface, then bisect that last step.
    let step = 0.5;
    let mut far = (1..2000).map(|i| i as f32 * step).find(|&t| below(t))?;
    let mut near = far - step;
    for _ in 0..16 {
        let middle = (near + far) / 2.0;
        if below(middle) { far = middle } else { near = middle }
    }

    let hit = origin + direction * far;
    contains(hit.x, hit.z).then_some(hit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raycast_lands_on_the_surface() {
        let hit = raycast(glam::vec3(10.0, 50.0, -20.0), glam::vec3(0.3, -1.0, 0.2).normalize())
            .expect("ray pointing down must hit the field");
        assert!((hit.y - height(hit.x, hit.z)).abs() < 0.01);
        assert!(raycast(glam::vec3(0.0, 50.0, 0.0), glam::Vec3::Y).is_none());
    }

    #[test]
    fn paths_are_where_the_formula_says() {
        // On the first path's centre line the distance is zero; well off both it isn't.
        let z = 14.0 * (0.4f32).sin() + 6.0 * (2.0f32).sin() - 9.0;
        assert!(path_distance(0.0, z) < 1e-4);
        assert!(path_distance(60.0, 60.0) > 5.0);
        assert!((0..100).all(|i| (0.0..=1.0).contains(&fbm(vec2(i as f32 * 0.37, i as f32 * 1.3)))));
    }
}
