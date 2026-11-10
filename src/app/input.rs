use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// What the player is doing with keyboard and mouse, in a form game code can
/// poll from a fixed-rate update instead of reacting to window events.
#[derive(Default)]
pub struct Input {
    held: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>,
    /// Cursor position in physical pixels.
    pub cursor: glam::Vec2,
    /// Left button went down on the world (not on UI) since the last step.
    pub clicked: bool,
    /// Left button is down on the world right now.
    pub left_held: bool,
}

impl Input {
    pub fn key(&mut self, code: KeyCode, down: bool) {
        if !down {
            self.held.remove(&code);
        } else if self.held.insert(code) {
            self.pressed.insert(code);
        }
    }

    pub fn held(&self, code: KeyCode) -> bool {
        self.held.contains(&code)
    }

    /// True for the one step in which the key went down.
    pub fn pressed(&self, code: KeyCode) -> bool {
        self.pressed.contains(&code)
    }

    /// -1, 0 or 1 from a pair of opposing keys.
    pub fn axis(&self, negative: KeyCode, positive: KeyCode) -> f32 {
        self.held(positive) as i8 as f32 - self.held(negative) as i8 as f32
    }

    /// Forget one-shot events once a step has seen them. Called after each
    /// fixed update, so a click on a frame with no update isn't lost.
    pub fn end_step(&mut self) {
        self.pressed.clear();
        self.clicked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressed_fires_once_and_ignores_key_repeat() {
        let mut input = Input::default();
        input.key(KeyCode::KeyW, true);
        assert!(input.pressed(KeyCode::KeyW) && input.held(KeyCode::KeyW));
        input.end_step();
        input.key(KeyCode::KeyW, true); // OS key repeat
        assert!(!input.pressed(KeyCode::KeyW) && input.held(KeyCode::KeyW));
        assert_eq!(input.axis(KeyCode::KeyS, KeyCode::KeyW), 1.0);
        input.key(KeyCode::KeyW, false);
        assert_eq!(input.axis(KeyCode::KeyS, KeyCode::KeyW), 0.0);
    }
}
