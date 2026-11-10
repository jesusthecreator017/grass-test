pub type Result<T> = std::result::Result<T, AppError>;

/// Everything that can go wrong because of the environment: no GPU, no window,
/// a missing asset. Programmer mistakes stay panics.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("failed to create the window")]
    Window(#[from] winit::error::OsError),

    #[error("event loop failure")]
    EventLoop(#[from] winit::error::EventLoopError),

    #[error("failed to create a surface for the window")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),

    #[error("no suitable GPU adapter")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),

    #[error("failed to get a device from the adapter")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    #[error("the adapter does not support this surface")]
    SurfaceUnsupported,

    #[error("failed to load image {path}")]
    LoadImage {
        path: String,
        #[source]
        source: image::ImageError,
    },

    #[error("failed to load model {path}")]
    LoadModel {
        path: String,
        #[source]
        source: gltf::Error,
    },

    #[error("model {path} is not supported: {reason}")]
    ModelUnsupported { path: String, reason: String },
}
