# Project Structure

```text
tile_compile/
├── web_frontend_v3/        # GUI3 HTML/CSS/JS frontend
├── web_backend_cpp/        # GUI3 Crow/C++ backend
├── tile_compile_cpp/
│   ├── apps/                # runner/cli entry points
│   ├── include/tile_compile/
│   ├── src/
│   ├── examples/            # example configs
│   ├── scripts/             # helper scripts
│   ├── tests/
│   ├── tile_compile.yaml
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── packaging/gui3/          # GUI3 release launchers/bundle helpers
├── docker/                  # Docker build/runtime images
├── docs/
│   ├── v3/                  # methodology docs
│   └── process_flow/        # implementation process-flow docs
├── start_backend.sh         # dev start for Crow backend + GUI3
├── README.md
└── README_de.md
```

## Active Components

| Component | Directory | Status | Stack |
|-----------|-----------|--------|-------|
| Core pipeline | `tile_compile_cpp/` | Active | C++20 + Eigen + OpenCV + cfitsio + yaml-cpp |
| GUI3 backend | `web_backend_cpp/` | Active | Crow + C++20 |
| GUI3 frontend | `web_frontend_v3/` | Active | HTML + CSS + JavaScript (ESM) |
