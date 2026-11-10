# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
cargo run                          # debug build; shader hot reload is on
cargo run --release                # no shader hot reload
cargo test                         # all tests (inline #[cfg(test)] modules, no tests/ dir)
cargo test shader_terrain_matches  # one test, by name substring
cargo clippy
```

- `CAPTURE=shot.png cargo run` renders one frame (frame 240 by default) to a PNG instead of the screen, then exits. `CAPTURE_FRAME=n` picks the frame. The window does not take focus in this mode. Use it to check rendering changes without a person looking at the window.
- Logging: default filter is `warn,grass_test=info`; `RUST_LOG` overrides it.

## Architecture

A wgpu + winit sandbox: a rolling grass field with a player cube, click-placed towers, and foxes that pathfind. Binary crate only.

**Engine / game split.** `src/app/` is the engine, `src/game.rs` is the only game-specific logic, `src/flora.rs` is the game's procedural content (plant and rock meshes plus placement, all built in code, no model files). Game code reaches the engine only through `game::Context` (gpu, renderer, input, orbit camera, audio), built by the `context!` macro in `app.rs`. It is a macro so the borrow checker sees `self.game` untouched.

**Frame loop** (`app.rs`, `State::frame`). Fixed 60 Hz steps drained from an accumulator, then one draw. `Input::end_step` clears edge-triggered state after each step. `editor.playing == false` is edit mode: game updates stop, clicks select entities or paint grass density, the camera still moves through `Game::camera`. Startup errors cannot return from `resumed()`, so they are parked on `App.error` and collected in `main::run`.

**ECS.** `hecs::World` lives on `Game`. An entity with `Transform` + `Renderable` is drawn; adding `Animator` makes it skinned; `GrassPusher` flattens grass (first 8 only); `Name` is what the editor lists. `Renderer::prepare` sorts entities by (mesh, texture, skinned entity id) and merges them into instanced batches.

**Rendering** (`renderer.rs`). Two passes per frame: shadow map from the light, then the 4x MSAA main pass resolving to the target. Draw order in the main pass: entities, ground, props, grass. Three independent subsystems, each with its own shader file and pipelines:

- `shader.wgsl` + `renderer.rs`: ECS entities. Bind groups: 0 texture, 1 camera, 2 light, 3 joints (`no_joints` is bound for unskinned meshes). The shadow pass binds the light's view in the camera slot.
- `grass.wgsl` + `grass.rs`: a compute shader fills fixed-capacity blade buffers (near and far level of detail) every frame; the same file also draws the ground. Density comes from a 512x512 painted map.
- `props.wgsl` + `props.rs`: scenery (plants, rocks) as a few meshes instanced thousands of times, grouped into 12.5 m chunks and culled per chunk. Props are not entities. `props.rs` also holds `MeshBuilder`/`Ribbon`, the mesh-building tools `flora.rs` uses.

**Terrain** (`terrain.rs`) is an analytic function of (x, z), not data. Ground mesh, object placement, mouse picking (`terrain::raycast`), and pathfinding all call it. `nav::Grid` is 1x1 m cells on XZ, A* with no corner cutting.

## Things that must stay in sync

- `terrain::height` and `terrain::path_distance` are duplicated in `grass.wgsl` (`terrain_height`, `path_distance`). The test `shader_terrain_matches_rust_terrain` compares the numeric literals of both in order, so change both together and keep the literals in the same order.
- `GrassUniform` in `grass.rs` mirrors the WGSL struct; a size test guards it.
- Vertex and instance layouts (`vertex.rs`, `instance.rs`, `PropVertex`/`PropInstance`) mirror shader inputs.

## Conventions

- `error::AppError` is for environment failures (no GPU, missing asset). Programmer mistakes stay panics.
- Shader hot reload (debug builds): saving any `.wgsl` rebuilds its pipelines via `try_build`, which uses a wgpu error scope so a bad shader is reported in the editor and the last good pipeline stays. Shaders are also compiled in with `include_wgsl!`, so release builds do not read them from disk.
- Asset and shader paths are absolute, from `env!("CARGO_MANIFEST_DIR")`. The binary is tied to this checkout.
- The editor's Save button writes `assets/editor.json` (grass and light settings) and `assets/grass_density.png`; both load at startup and are optional. The settings structs use `#[serde(default)]`, so new fields need a default, not a migration.
- Comments starting with `// ponytail:` mark a deliberate shortcut with its known ceiling and upgrade path. Read them before "fixing" a limitation; add one when taking a similar shortcut.
- Editor: F1 toggles it. Controls: WASD move, click build, right-drag or Q/E turn, scroll zoom, arrows pan, F follow.
