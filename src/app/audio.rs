use rodio::Source;
use std::time::Duration;

/// Sound output. A machine with no audio device still runs the game; it just
/// stays silent.
pub struct Audio {
    sink: Option<rodio::MixerDeviceSink>,
}

impl Audio {
    pub fn new() -> Self {
        let sink = rodio::DeviceSinkBuilder::open_default_sink()
            .inspect_err(|err| log::warn!("no audio output: {err}"))
            .ok();
        Self { sink }
    }

    /// A short tone - a stand-in until there are real sound files.
    pub fn beep(&self, frequency: f32) {
        let Some(sink) = &self.sink else { return };
        let tone = rodio::source::SineWave::new(frequency)
            .amplify(0.15)
            .take_duration(Duration::from_millis(90));
        sink.mixer().add(tone);
    }

    /// Plays a wav or ogg file to the end, mixed over whatever else is playing.
    #[allow(dead_code)] // no sound assets yet
    pub fn play_file(&self, path: &str) {
        let Some(sink) = &self.sink else { return };
        let played = std::fs::File::open(path)
            .map_err(|err| err.to_string())
            .and_then(|file| {
                rodio::play(sink.mixer(), std::io::BufReader::new(file))
                    .map_err(|err| err.to_string())
            });
        match played {
            Ok(player) => player.detach(),
            Err(err) => log::warn!("can't play {path}: {err}"),
        }
    }
}
