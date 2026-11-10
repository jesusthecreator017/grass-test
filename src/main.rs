mod app;
mod error;
mod flora;
mod game;

use app::App;
use error::Result;
use std::process::ExitCode;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> ExitCode {
    // Our own info! lines are on by default; wgpu's chatter stays at warn.
    // RUST_LOG overrides both.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,grass_test=info"),
    )
    .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            log::error!("{err}");
            if let Some(source) = std::error::Error::source(&err) {
                log::error!("caused by: {source}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    // resumed() can't return a Result, so a startup failure is parked on App
    // and collected here.
    app.take_error().map_or(Ok(()), Err)
}
