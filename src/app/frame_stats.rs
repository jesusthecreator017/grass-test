use std::time::{Instant, Duration};

pub struct FrameStats {
    last: Instant,
    elapsed: Duration,
    frames: u32,
    text: String,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            last: Instant::now(),
            elapsed: Duration::ZERO,
            frames: 0,
            text: String::from("-- fps"),
        }
    }
}

impl FrameStats {
    const WINDOW: Duration = Duration::from_millis(500);

    pub fn tick(&mut self) -> &str {
        let now = Instant::now();
        self.elapsed += now - self.last;
        self.last = now;
        self.frames += 1;

        if self.elapsed > Self::WINDOW {
            let seconds = self.elapsed.as_secs_f32();
            let fps = self.frames as f32 / seconds;
            let frame_ms = seconds * 1000.0 / self.frames as f32;
            self.text = format!("{fps:.0} fps {frame_ms:.2} ms");
            self.elapsed = Duration::ZERO;
            self.frames = 0;
        }

        &self.text
    }
}
