# Tauri + Vanilla

This template should help get you started developing with Tauri in vanilla HTML, CSS and Javascript.

## Prerequisites

To run `tauri dev` / `tauri build`, you need:

- **Node.js + npm**
- **Rust toolchain (cargo)** available in your PATH
  - `cargo` is required because Tauri builds the Rust backend.

If you see an error like:

```text
failed to run 'cargo metadata' ... No such file or directory
```

then Rust/Cargo is missing.

### Rust installation hints (manual)

- **Linux/macOS:** install Rust via `rustup` (recommended by Rust)
  - https://rustup.rs/
- **Windows:** install Rust via `rustup` (MSVC toolchain)
  - https://rustup.rs/

After installation, verify:

```text
cargo --version
rustc --version
```

## Development

From this `gui/` directory:

```text
npm install
npm run dev
```

## Usage (Scan → Confirm → Run)

1. Use **Scan** to analyze the input directory.
2. If the scan reports `color_mode = "UNKNOWN"` (e.g. missing FITS header `BAYERPAT`), you must confirm the color mode once.
3. Only then **Start** is enabled.

The confirmed value is passed as `color_mode_confirmed` and is written into the run folder:

- `runs/<ts>_<run_id>/run_metadata.json`
- `runs/<ts>_<run_id>/logs/run_events.jsonl` (`run_start.color_mode_confirmed`)

## Build

```text
npm run build
```

## Troubleshooting

### Linux (Wayland): `Gdk-Message ... Error 71 (Protocol error) dispatching to Wayland display`

This is a runtime issue in the GTK/GDK stack when running under Wayland on some setups.

Workarounds:

1. **Force X11 backend** for the dev session:

```text
GDK_BACKEND=x11 npm run dev
```

2. If needed, also force winit to use X11:

```text
GDK_BACKEND=x11 WINIT_UNIX_BACKEND=x11 npm run dev
```

3. Alternative: start an **X11/XWayland session** (desktop/login setting) and run `npm run dev` there.

### Linux (GPU/GBM): `Failed to create GBM buffer ... invalid argument`

This is typically related to GPU/driver/DMABUF/GBM paths in the WebView stack.

Workarounds:

1. Disable dmabuf renderer (WebKitGTK):

```text
WEBKIT_DISABLE_DMABUF_RENDERER=1 npm run dev
```

2. If it still crashes, force software rendering:

```text
WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1 GDK_GL=disable npm run dev
```

Notes:

- The repo root `start_gui.sh` applies `WEBKIT_DISABLE_DMABUF_RENDERER=1` by default.
- You can enable software rendering via:

```text
TILE_COMPILE_GUI_SOFTWARE=1 ./start_gui.sh
```

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
