use pathfinding::prelude::astar;
use std::collections::HashSet;

pub type Cell = (i32, i32);

/// A square grid of 1x1 cells on the XZ plane, centred on the origin, that
/// knows which cells are built on.
pub struct Grid {
    half: i32,
    blocked: HashSet<Cell>,
}

impl Grid {
    pub fn new(half: i32) -> Self {
        Self { half, blocked: HashSet::new() }
    }

    pub fn cell_of(position: glam::Vec3) -> Cell {
        (position.x.floor() as i32, position.z.floor() as i32)
    }

    /// The (x, z) of a cell's centre.
    pub fn center(cell: Cell) -> (f32, f32) {
        (cell.0 as f32 + 0.5, cell.1 as f32 + 0.5)
    }

    pub fn is_free(&self, cell: Cell) -> bool {
        let inside = |v: i32| (-self.half..self.half).contains(&v);
        inside(cell.0) && inside(cell.1) && !self.blocked.contains(&cell)
    }

    pub fn block(&mut self, cell: Cell) {
        self.blocked.insert(cell);
    }

    pub fn clear(&mut self) {
        self.blocked.clear();
    }

    /// Shortest walkable route, both ends included. Diagonal steps are allowed
    /// only when they don't squeeze past a blocked corner.
    pub fn path(&self, start: Cell, goal: Cell) -> Option<Vec<Cell>> {
        let neighbours = |&(x, z): &Cell| {
            let mut out = Vec::with_capacity(8);
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if self.is_free((x + dx, z + dz)) {
                    out.push(((x + dx, z + dz), 10));
                }
            }
            for (dx, dz) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                if self.is_free((x + dx, z + dz))
                    && self.is_free((x + dx, z))
                    && self.is_free((x, z + dz))
                {
                    out.push(((x + dx, z + dz), 14));
                }
            }
            out
        };
        let heuristic = |&(x, z): &Cell| {
            let (dx, dz) = ((x - goal.0).abs(), (z - goal.1).abs());
            10 * dx.max(dz) + 4 * dx.min(dz)
        };

        if !self.is_free(goal) {
            return None;
        }
        astar(&start, neighbours, heuristic, |&cell| cell == goal).map(|(path, _)| path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_goes_around_a_wall() {
        let mut grid = Grid::new(10);
        // A wall across x = 0 with a single gap at z = 5.
        for z in -10..10 {
            if z != 5 {
                grid.block((0, z));
            }
        }
        let path = grid.path((-3, 0), (3, 0)).expect("gap makes it reachable");
        assert!(path.contains(&(0, 5)));
        assert!(path.iter().all(|&cell| grid.is_free(cell)));

        grid.block((0, 5));
        assert!(grid.path((-3, 0), (3, 0)).is_none());
    }
}
