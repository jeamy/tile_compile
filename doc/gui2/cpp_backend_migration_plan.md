# C++ Backend Stand: Python-Eliminierung abgeschlossen

## Ziel

Dokumentation des abgeschlossenen Crow/C++-Backendstands nach vollständiger Eliminierung der produktiven Python-Abhängigkeiten:

- `web_backend_cpp/` ist der produktive HTTP/WebSocket-Server auf Basis von **Crow**
- Stats/Reports werden über den integrierten C++-Pfad erzeugt
- Python bleibt nur noch als optionales Entwicklungswerkzeug fuer Analyse-/Hilfsskripte

Ergebnis: Eine einzelne auslieferbare Binary pro Plattform — kein Python, kein venv, kein pip.

---

## Framework-Entscheidung: Crow

### Warum Crow (nicht Drogon)

| Kriterium | Crow | Drogon |
|---|---|---|
| Header-only | ✅ | ❌ (libdrogon, trantor, Brotli, ...) |
| Windows | ✅ problemlos | ⚠️ notorisch schwierige Windows-Builds |
| macOS AppBundle | ✅ (keine Dylib-Abhängigkeiten) | ⚠️ zieht OpenSSL, libpq, redis etc. rein |
| nlohmann/json | ✅ (bereits im Projekt!) | ❌ (eigenes JSON) |
| yaml-cpp | ✅ (bereits im Projekt!) | ❌ |
| WebSocket nativ | ✅ | ✅ |
| Static file serving | ✅ eingebaut | ✅ eingebaut |
| CMake FetchContent | ✅ trivial | ⚠️ komplexeres Build-System |
| Statisch linkbar | ✅ eine Binary | ⚠️ viele Shared Libs |

Das Backend ist **kein Hochlast-API-Server** (max. ein paar gleichzeitige Runner-Jobs).
Drogons Performance-Vorteile sind hier irrelevant.

### Crow-Version

`v1.2.0` — stabile Version mit vollem WebSocket-Support.

```cmake
FetchContent_Declare(Crow
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG        v1.2.0)
FetchContent_MakeAvailable(Crow)
```

Crow benötigt nur **Asio** (Boost.Asio oder standalone Asio).
Standalone Asio ist ebenfalls per FetchContent beziehbar.

---

## Projekstruktur: Neues Verzeichnis `web_backend_cpp/`

```
tile_compile/
├── web_backend_cpp/          ← produktives C++ Backend
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp                    ← Crow app, Port, Startup
│   │   ├── app_state.hpp               ← AppState-Struct (shared state)
│   │   ├── backend_runtime.hpp/.cpp    ← Pfad-Auflösung, Security Policy
│   │   ├── job_store.hpp/.cpp          ← InMemoryJobStore (thread-safe)
│   │   ├── subprocess.hpp/.cpp         ← Subprocess-Management (Popen-Äquivalent)
│   │   ├── ui_event_store.hpp/.cpp     ← UiEventStore (JSONL append)
│   │   ├── config_revisions.hpp/.cpp   ← Config-Revision-Cache
│   │   ├── routes/
│   │   │   ├── system.cpp              ← GET /api/health, /api/version, /api/fs/*
│   │   │   ├── app_state_routes.cpp    ← GET /api/app/state, /api/app/constants
│   │   │   ├── config_routes.cpp       ← GET/POST /api/config/*
│   │   │   ├── scan_routes.cpp         ← POST /api/scan, GET /api/scan/*
│   │   │   ├── runs_routes.cpp         ← GET/POST /api/runs/*
│   │   │   ├── jobs_routes.cpp         ← GET/POST /api/jobs/*
│   │   │   ├── tools_routes.cpp        ← POST /api/tools/*
│   │   │   └── ws_routes.cpp           ← WS /api/ws/runs/{id}, /api/ws/jobs/{id}
│   │   └── report/
│   │       ├── report_generator.hpp/.cpp  ← integrierte Report-Erzeugung
│   │       ├── chart_svg.hpp/.cpp         ← SVG-Charts (kein matplotlib)
│   │       └── html_writer.hpp/.cpp       ← HTML/CSS output
│   └── tests/
│       └── ...
└── tile_compile_cpp/
    ├── apps/
    ├── build/
    └── scripts/
```

---

## Teil 1: C++ Crow-Backend

### 1.1 AppState (`app_state.hpp`)

Zentraler Shared State des produktiven Crow-Backends:

```cpp
struct AppState {
    std::shared_ptr<JobStore>         job_store;
    std::shared_ptr<BackendRuntime>   runtime;
    std::shared_ptr<CommandPolicy>    command_policy;
    std::shared_ptr<UiEventStore>     ui_event_store;
    std::shared_ptr<ConfigRevisions>  config_revisions;
    std::atomic<std::string*>         current_run_id;
    std::string                       last_scan_input_path;
    std::string                       active_config_revision_id;
    // Thread-safe accessors ...
};
```

### 1.2 BackendRuntime (`backend_runtime.hpp/.cpp`)

1:1 Übersetzung von `BackendRuntime` aus `command_runner.py`:

```cpp
struct BackendRuntime {
    std::filesystem::path project_root;
    std::filesystem::path cli_path;
    std::filesystem::path runner_path;
    std::filesystem::path runs_dir;
    std::filesystem::path default_config_path;
    std::vector<std::filesystem::path> allowed_roots;
    std::vector<std::filesystem::path> input_search_roots;

    static BackendRuntime autodetect();

    // Pfad-Validierung (Security Policy)
    std::filesystem::path ensure_path_allowed(
        const std::filesystem::path& path,
        bool must_exist = false,
        std::string_view label = "path") const;

    std::filesystem::path resolve_run_dir(
        std::string_view run_id_or_path,
        std::string_view runs_dir_override = "") const;

    std::filesystem::path resolve_runs_dir(
        std::string_view override = "") const;

    std::filesystem::path resolve_input_path(
        const std::filesystem::path& path,
        bool must_exist = false,
        std::string_view label = "input_path") const;
};
```

Umgebungsvariablen: `TILE_COMPILE_CLI`, `TILE_COMPILE_RUNNER`, `TILE_COMPILE_RUNS_DIR`,
`TILE_COMPILE_CONFIG_PATH`, `TILE_COMPILE_ALLOWED_ROOTS`, `TILE_COMPILE_INPUT_SEARCH_ROOTS`.

### 1.3 JobStore (`job_store.hpp/.cpp`)

1:1 Übersetzung von `InMemoryJobStore` aus `process_manager.py`:

```cpp
struct Job {
    std::string                         job_id;       // UUID
    std::string                         job_type;
    std::string                         state;        // pending|running|ok|error|cancelled
    nlohmann::json                      data;
    int                                 pid = -1;
    int                                 exit_code = -1;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::optional<std::chrono::system_clock::time_point> started_at;
    std::optional<std::chrono::system_clock::time_point> ended_at;
    std::shared_ptr<subprocess::Process> process;     // nullable
};

class JobStore {
public:
    Job create(std::string_view job_type, nlohmann::json data);
    std::optional<Job> get(std::string_view job_id) const;
    std::vector<Job> list() const;
    void set_state(std::string_view job_id, std::string_view state);
    void set_process(std::string_view job_id, std::shared_ptr<Process> proc);
    void set_exit_code(std::string_view job_id, int code);
    void merge_data(std::string_view job_id, const nlohmann::json& patch);
    std::optional<Job> cancel(std::string_view job_id);
    void clear_process(std::string_view job_id);
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Job> jobs_;
};
```

### 1.4 Subprocess-Management (`subprocess.hpp/.cpp`)

Plattformübergreifend: Linux/macOS via `fork/execvp`, Windows via `CreateProcess`.

```cpp
class Process {
public:
    static std::shared_ptr<Process> launch(
        const std::vector<std::string>& command,
        const std::filesystem::path& cwd,
        const std::optional<std::string>& stdin_text = std::nullopt);

    void terminate();               // SIGTERM / TerminateProcess
    bool is_running() const;
    int pid() const;

    // Blocking: liest stdout/stderr, gibt exit_code zurück
    int wait(std::string& stdout_out, std::string& stderr_out);

    // Non-blocking background launch (gibt Thread-Handle zurück)
    static void launch_background(
        std::shared_ptr<JobStore> job_store,
        const std::string& job_id,
        const std::vector<std::string>& command,
        const std::filesystem::path& cwd,
        const std::optional<std::string>& stdin_text = std::nullopt);
};
```

**Alternative**: `boost::process` falls Boost ohnehin als Abhängigkeit vorhanden ist.
Vorzugsweise eigene Implementierung um Abhängigkeiten minimal zu halten.

### 1.5 UiEventStore (`ui_event_store.hpp/.cpp`)

```cpp
class UiEventStore {
public:
    explicit UiEventStore(const std::filesystem::path& jsonl_path);

    void record(const nlohmann::json& event);  // append to JSONL + in-memory
    std::vector<nlohmann::json> list(int after_seq = 0, int limit = 200) const;
    int latest_seq() const;
private:
    mutable std::mutex mutex_;
    std::filesystem::path path_;
    std::vector<nlohmann::json> events_;  // in-memory cache
    int seq_ = 0;
};
```

### 1.6 Crow-Routen: Mapping Python → C++

#### Crow-Route-Syntax

```cpp
CROW_ROUTE(app, "/api/health").methods(crow::HTTPMethod::GET)
([&](const crow::request&) {
    return crow::response{crow::json::wvalue{{"status", "ok"}}};
});
```

#### Vollständige Routen-Tabelle

| Python-Datei | Endpoint | Methode | C++ Route-Datei |
|---|---|---|---|
| `system.py` | `/api/health` | GET | `system.cpp` |
| `system.py` | `/api/version` | GET | `system.cpp` |
| `system.py` | `/api/fs/roots` | GET | `system.cpp` |
| `system.py` | `/api/fs/list` | GET | `system.cpp` |
| `system.py` | `/api/fs/grant-root` | POST | `system.cpp` |
| `system.py` | `/api/fs/open` | POST | `system.cpp` |
| `app_state.py` | `/api/app/state` | GET | `app_state_routes.cpp` |
| `app_state.py` | `/api/app/constants` | GET | `app_state_routes.cpp` |
| `app_state.py` | `/api/app/ui-events` | GET | `app_state_routes.cpp` |
| `config.py` | `/api/config/schema` | GET | `config_routes.cpp` |
| `config.py` | `/api/config/current` | GET | `config_routes.cpp` |
| `config.py` | `/api/config/validate` | POST | `config_routes.cpp` |
| `config.py` | `/api/config/save` | POST | `config_routes.cpp` |
| `config.py` | `/api/config/presets` | GET | `config_routes.cpp` |
| `config.py` | `/api/config/presets/apply` | POST | `config_routes.cpp` |
| `config.py` | `/api/config/revisions` | GET | `config_routes.cpp` |
| `config.py` | `/api/config/revisions/{id}/restore` | POST | `config_routes.cpp` |
| `config.py` | `/api/config/patch` | POST | `config_routes.cpp` |
| `scan.py` | `/api/scan` | POST | `scan_routes.cpp` |
| `scan.py` | `/api/scan/quality` | GET | `scan_routes.cpp` |
| `scan.py` | `/api/scan/latest` | GET | `scan_routes.cpp` |
| `scan.py` | `/api/guardrails` | GET | `scan_routes.cpp` |
| `runs.py` | `/api/runs` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/status` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/logs` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/artifacts` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/artifacts/view` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/artifacts/raw/{path}` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/start` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/resume` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/stop` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/set-current` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/delete` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/stats` | POST | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/stats/status` | GET | `runs_routes.cpp` |
| `runs.py` | `/api/runs/{id}/config-revisions/{rid}/restore` | POST | `runs_routes.cpp` |
| `jobs.py` | `/api/jobs` | GET | `jobs_routes.cpp` |
| `jobs.py` | `/api/jobs/{id}` | GET | `jobs_routes.cpp` |
| `jobs.py` | `/api/jobs/{id}/cancel` | POST | `jobs_routes.cpp` |
| `tools.py` | `/api/tools/astrometry/detect` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/astrometry/install-cli` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/astrometry/catalog/download` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/astrometry/solve` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/astrometry/save-solved` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/pcc/siril/status` | GET | `tools_routes.cpp` |
| `tools.py` | `/api/tools/pcc/siril/download-missing` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/pcc/check-online` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/pcc/run` | POST | `tools_routes.cpp` |
| `tools.py` | `/api/tools/pcc/save-corrected` | POST | `tools_routes.cpp` |
| `ws.py` | `/api/ws/runs/{id}` | WS | `ws_routes.cpp` |
| `ws.py` | `/api/ws/jobs/{id}` | WS | `ws_routes.cpp` |
| `ws.py` | `/api/ws/system` | WS | `ws_routes.cpp` |

**Static files** (`/ui/*`): Crow's eingebauter Static-File-Handler.

#### WebSocket-Implementierung

Crow unterstützt WebSockets nativ via `crow::websocket`:

```cpp
CROW_ROUTE(app, "/api/ws/runs/<string>")
    .websocket()
    .onopen([&](crow::websocket::connection& conn, const std::string& run_id) {
        // Poll loop in separate thread
        std::thread([&conn, run_id, &app_state]() {
            int cursor = 0;
            while (!conn.is_close_initiated()) {
                auto events = tail_run_stream_events(run_dir, cursor);
                for (auto& ev : events) conn.send_text(ev.dump());
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }).detach();
    })
    .onclose([&](crow::websocket::connection&, const std::string&) {});
```

### 1.7 YAML-Verarbeitung

Python nutzt `yaml.safe_load` / `yaml.safe_dump` in `config.py` und `config/patch`.
**yaml-cpp** ist bereits im tile_compile_cpp Build vorhanden:

```cpp
#include <yaml-cpp/yaml.h>

// Load
YAML::Node config = YAML::Load(yaml_text);

// Dotted path set (entspricht _set_dotted() in config.py)
void set_dotted(YAML::Node& root, const std::string& path, const YAML::Node& value);

// Dump back to string
YAML::Emitter out;
out << config;
std::string yaml_out = out.c_str();
```

### 1.8 Fehlerbehandlung / HTTP-Fehler

Python nutzt `HTTPException` mit strukturierten Fehler-Bodies.
In Crow:

```cpp
crow::response make_error(int status, std::string_view code, std::string_view message,
                          nlohmann::json details = nullptr) {
    crow::response res{status};
    res.set_header("Content-Type", "application/json");
    nlohmann::json body = {{"error", {{"code", code}, {"message", message}}}};
    if (!details.is_null()) body["error"]["details"] = details;
    res.body = body.dump();
    return res;
}
```

### 1.9 Download-Manager (`tools_routes.cpp`)

Die Download-Logik in `tools.py` (`download_file_with_retry`, Resume-Support) wird in C++ mit `libcurl` oder `cpp-httplib` (already used?) implementiert:

- **libcurl** ist cross-platform und unterstützt Resume (`CURLOPT_RESUME_FROM`), Redirect-Following, Timeout
- Falls `cpp-httplib` bereits im Projekt vorhanden: als Alternative prüfen

```cpp
class FileDownloader {
public:
    struct Options {
        int timeout_s = 1800;
        int retry_count = 2;
        double retry_backoff_s = 1.5;
        bool resume = true;
    };

    using ProgressCb = std::function<void(int64_t received, int64_t total)>;
    using StateCb    = std::function<void(const nlohmann::json& patch)>;

    void download(const std::string& url,
                  const std::filesystem::path& dest,
                  const Options& opts,
                  ProgressCb progress_cb,
                  StateCb state_cb);
};
```

### 1.10 main.cpp

```cpp
int main(int argc, char* argv[]) {
    auto state = std::make_shared<AppState>();
    state->runtime = BackendRuntime::autodetect();
    state->job_store = std::make_shared<JobStore>();
    state->command_policy = std::make_shared<CommandPolicy>(*state->runtime);
    state->ui_event_store = std::make_shared<UiEventStore>(
        state->runtime->project_root / "web_backend_cpp/runtime/ui_events.jsonl");
    state->config_revisions = std::make_shared<ConfigRevisions>();

    crow::App<crow::CORSHandler> app;

    // CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors.global().origin("*");

    // Register routes
    register_system_routes(app, state);
    register_app_state_routes(app, state);
    register_config_routes(app, state);
    register_scan_routes(app, state);
    register_runs_routes(app, state);
    register_jobs_routes(app, state);
    register_tools_routes(app, state);
    register_ws_routes(app, state);

    // Static files: /ui/ → web_frontend/
    auto frontend_dir = state->runtime->project_root / "web_frontend";
    if (std::filesystem::exists(frontend_dir)) {
        CROW_ROUTE(app, "/ui/<path>")([&frontend_dir](const crow::request&, std::string path) {
            return crow::response{crow::StaticDir{frontend_dir}};
        });
    }

    int port = 8000;
    if (argc > 1) port = std::stoi(argv[1]);

    app.port(port).multithreaded().run();
    return 0;
}
```

---

## Teil 2: C++ Report-Generator

### 2.1 Kernproblem: matplotlib → SVG

Der fruehere Python-Reportpfad nutzte `matplotlib` für:
- Zeitreihen-Liniendiagramme
- Histogramme
- Scatter-Plots
- 2D-Heatmaps (Tile-Spatial-Maps)
- Balkendiagramme (horizontal + vertikal)
- Kreisdiagramme (Pie-Charts)

**Lösung**: SVG-Charts direkt in C++ generieren — kein externer Chart-Renderer nötig.
SVG ist direkt in HTML einbettbar und benötigt keinen Browser-Plugin.

Die vormals im Python-Reportpfad erzeugten PNGs werden durch **eingebettete SVGs** im HTML ersetzt.
Das vereinfacht den Output (kein `artifacts/*.png`, alles in `report.html`).

### 2.2 Chart-SVG-Modul (`chart_svg.hpp/.cpp`)

Jeder Chart-Typ wird als SVG-String zurückgegeben, der direkt in HTML eingebettet wird:

#### Zeitreihen-Liniendiagramm

```cpp
struct TimeseriesOptions {
    std::string color    = "#7aa2f7";
    bool        median_line = true;
    std::string xlabel   = "frame index";
    std::string ylabel;
    std::string title;
    int         width    = 700;
    int         height   = 300;
};

std::string svg_timeseries(
    const std::vector<double>& values,
    const TimeseriesOptions& opts);
```

Implementierung:
- Min/Max/Padding berechnen → Viewport-Koordinaten
- Polyline-SVG-Element für Linienverlauf
- Optionale gestrichelte Median-Linie
- Achsen mit einfachen Tick-Marks und Labels (SVG `<text>`)

#### Histogramm

```cpp
struct HistogramOptions {
    int         bins  = 60;
    std::string color = "#7aa2f7";
    std::string title;
    std::string xlabel;
    int         width = 550;
    int         height = 300;
};

std::string svg_histogram(
    const std::vector<double>& values,
    const HistogramOptions& opts);
```

Implementierung:
- 1%/99% Percentile clippen (wie Python-Version)
- Bin-Breite berechnen, Counts akkumulieren
- SVG `<rect>` pro Bin

#### Scatter-Plot

```cpp
struct ScatterOptions {
    std::string title;
    std::string xlabel;
    std::string ylabel;
    std::string colormap = "plasma";  // plasma|viridis|rainbow
    int         width    = 550;
    int         height   = 450;
};

std::string svg_scatter(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::optional<std::vector<double>>& color_values,
    const ScatterOptions& opts);
```

#### Spatial Tile Heatmap

```cpp
struct TileHeatmapOptions {
    std::string title;
    std::string label     = "value";
    std::string colormap  = "viridis";
    bool        show_grid = true;
    int         max_width = 800;   // Output-Breite in SVG-Einheiten
};

struct TileGeometry {
    int x, y, width, height;
};

std::string svg_spatial_tile_heatmap(
    const std::vector<TileGeometry>& tiles,
    const std::vector<double>& values,
    int img_w, int img_h,
    const TileHeatmapOptions& opts);
```

Implementierung:
- Colormap-Lookup (viridis/plasma/inferno als fest kodierte 256-Farb-Tabellen, identisch zu matplotlib)
- Pro Tile ein SVG `<rect>` mit `fill` = colormap(value)
- Optionales Tile-Grid als überlagerte Rechtecke mit `stroke`
- Colorbar als Gradient-SVG rechts

#### Balkendiagramm (horizontal + vertikal)

```cpp
std::string svg_bar(
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const std::string& title,
    const std::string& ylabel,
    const std::vector<std::string>& colors = {});

std::string svg_bar_horizontal(
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const std::string& title,
    const std::string& xlabel,
    const std::vector<std::string>& colors = {});
```

#### Multi-Zeitreihen

```cpp
std::string svg_multi_timeseries(
    const std::map<std::string, std::vector<double>>& series,
    const std::string& title,
    const std::string& ylabel);
```

#### BGE-Grid-Scatter

```cpp
std::string svg_bge_grid_scatter(
    const std::vector<nlohmann::json>& channels,
    const std::string& title);
```

#### Warp-Scatter (Registrierung)

```cpp
std::string svg_warp_scatter(
    const std::vector<nlohmann::json>& warps,
    const std::vector<double>& ccs,
    const std::string& title);
```

#### Pie-Chart

```cpp
std::string svg_pie(
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const std::vector<std::string>& colors,
    const std::string& title);
```

### 2.3 Colormap-Implementierung

Matplotlib's Colormaps werden als statische 256-Farb-RGB-Tabellen eingebettet
(aus matplotlib-Quellcode extrahierbar oder aus bekannten Werten hartcodiert):

```cpp
// In colormap_data.hpp
extern const uint8_t VIRIDIS_RGB[256][3];
extern const uint8_t PLASMA_RGB[256][3];
extern const uint8_t INFERNO_RGB[256][3];
extern const uint8_t MAGMA_RGB[256][3];
extern const uint8_t CIVIDIS_RGB[256][3];
extern const uint8_t YLGN_RGB[256][3];
extern const uint8_t YLGNBU_RGB[256][3];

std::string colormap_hex(const uint8_t table[256][3], double t);  // t in [0,1]
```

### 2.4 Report-Generator (`report_generator.hpp/.cpp`)

Direkte 1:1-Übersetzung der frueheren Python-Sektions-Generatoren:

```cpp
struct ReportSection {
    std::string title;
    std::string cards_html;  // inline SVG statt PNG-Referenzen
};

class ReportGenerator {
public:
    explicit ReportGenerator(const std::filesystem::path& run_dir);

    // Entspricht der frueheren Python-Reportlogik
    std::filesystem::path generate();

private:
    // Sektions-Generatoren
    ReportSection gen_timeline(const std::vector<nlohmann::json>& events);
    ReportSection gen_frame_usage(const std::vector<nlohmann::json>& events);
    ReportSection gen_normalization(const nlohmann::json& norm);
    ReportSection gen_global_metrics(const nlohmann::json& gm);
    ReportSection gen_tile_grid(const nlohmann::json& tg);
    ReportSection gen_registration(const nlohmann::json& reg);
    ReportSection gen_local_metrics(const nlohmann::json& lm, const nlohmann::json& tg);
    ReportSection gen_reconstruction(const nlohmann::json& recon, const nlohmann::json& tg);
    ReportSection gen_clustering(const nlohmann::json& cl);
    ReportSection gen_synthetic(const nlohmann::json& syn);
    ReportSection gen_bge(const nlohmann::json& bge);
    ReportSection gen_validation(const nlohmann::json& val);
    ReportSection gen_common_overlap(const nlohmann::json& co);

    // HTML-Builder
    std::string make_card_html(
        const std::string& title,
        const std::vector<std::string>& svgs,
        const std::vector<std::string>& evals,
        const std::string& status = "",
        const std::map<int, std::string>& explanations = {});

    std::string make_chart_row(const std::string& svg, const std::string& explain_html);
    std::string infer_status(const std::vector<std::string>& evals);

    // Daten
    std::filesystem::path run_dir_;
    std::filesystem::path artifacts_dir_;

    // Statistik-Helfer (entspricht der frueheren Python-Logik, ohne numpy)
    struct BasicStats {
        int n = 0;
        double min = 0, max = 0, mean = 0, median = 0, std_dev = 0;
    };
    static BasicStats basic_stats(std::vector<double> vals);

    // HTML/CSS output
    void write_css(const std::filesystem::path& path);
    void write_html(
        const std::filesystem::path& path,
        const std::string& title,
        const std::vector<std::string>& meta_lines,
        const std::vector<ReportSection>& sections,
        const std::string& config_text = "");
};
```

### 2.5 CLI-Integration in `tile_compile_cli`

Der Report-Generator wird als neuer Subcommand in `tile_compile_cli` eingebaut:

```
tile_compile_cli generate-report /path/to/run_dir
```

Damit ist der fruehere Python-Aufruf vollstaendig durch `tile_compile_cli generate-report` ersetzt.
Das Backend ruft `runtime.cli_path generate-report <run_dir>` direkt auf.

---

## Teil 3: CMake-Integration

### 3.1 `web_backend_cpp/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(tile_compile_web_backend CXX)

set(CMAKE_CXX_STANDARD 20)

# Crow (header-only)
include(FetchContent)
FetchContent_Declare(Crow
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG        v1.2.0
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(Crow)

# Asio (standalone, kein Boost)
FetchContent_Declare(asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG        asio-1-30-2
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(asio)

# nlohmann/json (bereits im Hauptprojekt, falls extern verfügbar)
find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# yaml-cpp (bereits im Hauptprojekt)
find_package(yaml-cpp REQUIRED)

# libcurl (für Download-Manager)
find_package(CURL REQUIRED)

add_executable(tile_compile_web_backend
  src/main.cpp
  src/backend_runtime.cpp
  src/job_store.cpp
  src/subprocess.cpp
  src/ui_event_store.cpp
  src/config_revisions.cpp
  src/routes/system.cpp
  src/routes/app_state_routes.cpp
  src/routes/config_routes.cpp
  src/routes/scan_routes.cpp
  src/routes/runs_routes.cpp
  src/routes/jobs_routes.cpp
  src/routes/tools_routes.cpp
  src/routes/ws_routes.cpp
  src/report/report_generator.cpp
  src/report/chart_svg.cpp
  src/report/html_writer.cpp
)

target_link_libraries(tile_compile_web_backend
  PRIVATE
    Crow::Crow
    nlohmann_json::nlohmann_json
    yaml-cpp
    CURL::libcurl
)

target_include_directories(tile_compile_web_backend
  PRIVATE ${asio_SOURCE_DIR}/asio/include)
```

### 3.2 Integration in `tile_compile_cpp/CMakeLists.txt`

```cmake
# Optionaler Web-Backend-Build
option(BUILD_WEB_BACKEND "Build C++ web backend" OFF)
if (BUILD_WEB_BACKEND)
  add_subdirectory(../web_backend_cpp web_backend_cpp)
endif()
```

---

## Teil 4: Plattformspezifisches Packaging

### Linux (AppImage)

```
AppDir/
├── usr/bin/tile_compile_web_backend
├── usr/bin/tile_compile_runner
├── usr/bin/tile_compile_cli
└── AppRun
```

Crow ist statisch linkbar → eine Binary ohne externe `.so`-Abhängigkeiten.
Ausnahme: `libcurl` muss ggf. statisch gelinkt werden (`CURL_STATICLIB`).

### macOS (.app Bundle)

```
TileCompile.app/
└── Contents/
    ├── MacOS/
    │   ├── tile_compile_web_backend
    │   ├── tile_compile_runner
    │   └── tile_compile_cli
    ├── Frameworks/
    │   └── (nur libcurl falls shared)
    └── Resources/
        └── web_frontend/
```

`tile_compile_web_backend` öffnet automatisch den Browser auf `http://localhost:8000/ui/`
(via `open http://localhost:8000/ui/` auf macOS, `xdg-open` auf Linux, `ShellExecute` auf Windows).

### Windows (.exe)

```
tile_compile/
├── tile_compile_web_backend.exe
├── tile_compile_runner.exe
├── tile_compile_cli.exe
└── web_frontend/
```

Mit MSVC oder MinGW statisch linkbar.
Crow ist MSVC-kompatibel.

---

## Teil 5: Implementierungs-Reihenfolge

### Phase 1: Grundgerüst (Priorität: Hoch)

1. `web_backend_cpp/CMakeLists.txt` mit Crow/Asio/nlohmann/yaml-cpp
2. `backend_runtime.hpp/.cpp` — Pfad-Auflösung, Security Policy
3. `job_store.hpp/.cpp` — In-Memory Job Store
4. `subprocess.hpp/.cpp` — Plattformneutrale Prozessverwaltung
5. `main.cpp` — Crow App, Port, CORS
6. `routes/system.cpp` — `/api/health`, `/api/version`, `/api/fs/*`
7. `routes/jobs_routes.cpp` — `/api/jobs/*`
8. Build testen, Smoke-Test gegen laufendes Frontend

### Phase 2: Core-Routen (Priorität: Hoch)

9.  `routes/runs_routes.cpp` — Start, Stop, Status, Logs, Artifacts (größte Datei!)
10. `routes/scan_routes.cpp` — Scan, Quality, Guardrails
11. `routes/config_routes.cpp` — Schema, Current, Validate, Save, Patch, Presets, Revisions
12. `ui_event_store.hpp/.cpp` + `routes/app_state_routes.cpp`
13. `routes/ws_routes.cpp` — WebSocket für Run/Job/System

### Phase 3: Tools + Downloads (Priorität: Mittel)

14. Download-Manager (libcurl-basiert, Resume-Support)
15. `routes/tools_routes.cpp` — ASTAP, PCC, Siril

### Phase 4: Report-Generator (Priorität: Mittel)

16. `chart_svg.hpp/.cpp` — Alle Chart-Typen als SVG
17. Colormap-Tabellen (`colormap_data.hpp`)
18. `report/report_generator.cpp` — Alle Sektions-Generatoren
19. `tile_compile_cli generate-report` Subcommand
20. Backend ruft `generate-report` statt Python auf

### Phase 5: Packaging + Migration (Priorität: Hoch)

21. Build-Scripts anpassen (Linux AppImage, macOS .app, Windows .exe) — erledigt
22. Smoke-Tests und produktive Crow-Verifikation — erledigt
23. Alten Python-Backendpfad als deprecated markieren — erledigt
24. Python-Backend aus Default-Startup entfernen — erledigt
25. Alten Python-Backendpfad nach vollständiger Verifikation löschen — erledigt

---

## Teil 6: Bekannte Unterschiede / Entscheidungen

### YAML-Dump Format

Python's `yaml.safe_dump` hat spezifisches Verhalten (Quoting, Zeilenumbrüche, `null`-Darstellung).
yaml-cpp produziert leicht anderen Output. Das Frontend muss mit beiden Formaten umgehen können —
das ist bereits der Fall, da YAML nur gelesen/gespeichert und nicht geparst wird.

### Config-Patch (_set_dotted)

Python's `_set_dotted()` nutzt `yaml.safe_load` für Value-Parsing (`parse_values=True`).
In C++ wird der gleiche Effekt mit `YAML::Load(value_string)` erzielt.

### Popen-Semantik / Queue-Worker

Der Queue-Worker in `runs.py` läuft als synchroner Thread mit `proc.communicate()`.
In C++ wird das ebenfalls als `std::thread` + `Process::wait()` implementiert.

### Download-Resume

Python's `download_file_with_retry` implementiert HTTP-Resume mit
`Range: bytes=<offset>-` Header.
libcurl unterstützt das nativ via `CURLOPT_RESUME_FROM_LARGE`.

### Report: PNG → inline SVG

Die eingebetteten SVGs sind für den Use-Case (einmalig generierter Report)
vollständig äquivalent zu Matplotlib-PNGs. Inline SVG hat den Vorteil:
- Keine separaten PNG-Dateien im Artifacts-Verzeichnis
- Report ist eine einzelne HTML-Datei (einfacher zu teilen)
- Schriften/Farben exakt kontrollierbar
- Kein Image-Loading bei Offline-Betrachtung

Falls für spezifische Charts matplotlib-exakte Ausgabe benötigt wird
(z.B. für wissenschaftliche Reports), kann alternativ ein eingebetteter
Python-Less Chart-Renderer (wie `sciplot` oder `gnuplot` via Pipe) verwendet werden.
Das ist aber für diesen Use-Case nicht erforderlich.

---

## Teil 6: Frontend-Kompatibilität — Kein Umbau nötig

Das Frontend kommuniziert ausschließlich über `web_frontend/src/constants.js`.
**Alle dort definierten Endpoints werden 1:1 im C++ Backend implementiert — kein einziger wird umbenannt oder entfernt.**

### Vollständiger Abgleich `constants.js` ↔ C++ Backend

| `constants.js`-Key | Endpoint | C++ Route-Datei |
|---|---|---|
| `fs.grantRoot` | `POST /api/fs/grant-root` | `system.cpp` |
| `fs.openPath` | `POST /api/fs/open` | `system.cpp` |
| `fs.roots` *(implizit)* | `GET /api/fs/roots` | `system.cpp` |
| `fs.list` *(implizit)* | `GET /api/fs/list` | `system.cpp` |
| *(main.js L308)* | `GET /api/jobs` | `jobs_routes.cpp` |
| `jobs.byId` | `GET /api/jobs/{id}` | `jobs_routes.cpp` |
| `guardrails.root` | `GET /api/guardrails` | `scan_routes.cpp` |
| `app.state` | `GET /api/app/state` | `app_state_routes.cpp` |
| `app.constants` | `GET /api/app/constants` | `app_state_routes.cpp` |
| *(main.js L309)* | `GET /api/app/ui-events` | `app_state_routes.cpp` |
| `scan.root` | `POST /api/scan` | `scan_routes.cpp` |
| `scan.latest` | `GET /api/scan/latest` | `scan_routes.cpp` |
| `scan.quality` | `GET /api/scan/quality` | `scan_routes.cpp` |
| `config.current` | `GET /api/config/current` | `config_routes.cpp` |
| `config.patch` | `POST /api/config/patch` | `config_routes.cpp` |
| `config.presets` | `GET /api/config/presets` | `config_routes.cpp` |
| `config.applyPreset` | `POST /api/config/presets/apply` | `config_routes.cpp` |
| `config.validate` | `POST /api/config/validate` | `config_routes.cpp` |
| `config.save` | `POST /api/config/save` | `config_routes.cpp` |
| `config.revisions` | `GET /api/config/revisions` | `config_routes.cpp` |
| *(main.js L181)* | `POST /api/config/revisions/{id}/restore` | `config_routes.cpp` |
| `runs.list` | `GET /api/runs` | `runs_routes.cpp` |
| `runs.start` | `POST /api/runs/start` | `runs_routes.cpp` |
| `runs.status` | `GET /api/runs/{id}/status` | `runs_routes.cpp` |
| `runs.artifacts` | `GET /api/runs/{id}/artifacts` | `runs_routes.cpp` |
| `runs.artifactView` | `GET /api/runs/{id}/artifacts/view` | `runs_routes.cpp` |
| `runs.artifactRaw` | `GET /api/runs/{id}/artifacts/raw/{path}` | `runs_routes.cpp` |
| `runs.delete` | `POST /api/runs/{id}/delete` | `runs_routes.cpp` |
| `runs.stop` | `POST /api/runs/{id}/stop` | `runs_routes.cpp` |
| `runs.resume` | `POST /api/runs/{id}/resume` | `runs_routes.cpp` |
| `runs.stats` | `POST /api/runs/{id}/stats` | `runs_routes.cpp` |
| `runs.statsStatus` | `GET /api/runs/{id}/stats/status` | `runs_routes.cpp` |
| `runs.logs` | `GET /api/runs/{id}/logs?tail=N` | `runs_routes.cpp` |
| `runs.setCurrent` | `POST /api/runs/{id}/set-current` | `runs_routes.cpp` |
| `runs.restoreRevision` | `POST /api/runs/{id}/config-revisions/{rid}/restore` | `runs_routes.cpp` |
| `ws.run` | `WS /api/ws/runs/{id}` | `ws_routes.cpp` |
| `astrometry.detect` | `POST /api/tools/astrometry/detect` | `tools_routes.cpp` |
| `astrometry.installCli` | `POST /api/tools/astrometry/install-cli` | `tools_routes.cpp` |
| `astrometry.downloadCatalog` | `POST /api/tools/astrometry/catalog/download` | `tools_routes.cpp` |
| `astrometry.cancelDownload` | `POST /api/tools/astrometry/catalog/cancel` | `tools_routes.cpp` |
| `astrometry.solve` | `POST /api/tools/astrometry/solve` | `tools_routes.cpp` |
| `astrometry.saveSolved` | `POST /api/tools/astrometry/save-solved` | `tools_routes.cpp` |
| `pcc.sirilStatus` | `GET /api/tools/pcc/siril/status` | `tools_routes.cpp` |
| `pcc.downloadMissing` | `POST /api/tools/pcc/siril/download-missing` | `tools_routes.cpp` |
| `pcc.cancelDownload` | `POST /api/tools/pcc/siril/cancel` | `tools_routes.cpp` |
| `pcc.checkOnline` | `POST /api/tools/pcc/check-online` | `tools_routes.cpp` |
| `pcc.run` | `POST /api/tools/pcc/run` | `tools_routes.cpp` |
| `pcc.saveCorrected` | `POST /api/tools/pcc/save-corrected` | `tools_routes.cpp` |

**Ergebnis: Das Frontend (`constants.js`) muss nicht geändert werden.** Alle **49 Endpoints** werden mit identischer URL, HTTP-Methode und identischem JSON-Schema implementiert.

### Response-Schema-Treue

Crow gibt `crow::json::wvalue` bzw. `nlohmann::json` als Response zurück.
Alle Response-Strukturen werden exakt aus den Python-Rückgabewerten übernommen (gleiche Keys, gleiche Typen, gleiche Nesting-Tiefe). Das Frontend bemerkt keinen Unterschied.

---

## Teil 7: GitHub Actions CI/CD

### Workflow-Datei: `.github/workflows/build.yml`

Ziele:
- **Linux x86_64**: Ubuntu 22.04 (glibc 2.35 → läuft auf Ubuntu 20.04+, Debian 11+, Fedora 36+)
- **macOS**: macOS 13 (arm64/Apple Silicon) + macOS 12 (x86_64)
- **Windows**: Windows Server 2022, MSVC 2022
- **AppImage** (Linux): Automatisch aus dem Linux-Build erstellt

```yaml
name: Build

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  build-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            cmake ninja-build \
            libeigen3-dev \
            libopencv-dev \
            libyaml-cpp-dev \
            libcurl4-openssl-dev \
            libssl-dev \
            fuse libfuse2  # für AppImage-Tool

      - name: Build (Release)
        run: |
          cmake -B build -S tile_compile_cpp \
            -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_WEB_BACKEND=ON
          cmake --build build --target tile_compile_web_backend tile_compile_runner tile_compile_cli

      - name: Package AppImage
        run: |
          # linuxdeploy + AppImage-Builder
          wget -q https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
          chmod +x linuxdeploy-x86_64.AppImage
          mkdir -p AppDir/usr/bin AppDir/usr/share/tile_compile
          cp build/tile_compile_web_backend AppDir/usr/bin/
          cp build/tile_compile_runner      AppDir/usr/bin/
          cp build/tile_compile_cli         AppDir/usr/bin/
          cp -r web_frontend/dist           AppDir/usr/share/tile_compile/ui  # vite build
          cat > AppDir/AppRun << 'EOF'
          #!/bin/bash
          HERE="$(dirname "$(readlink -f "$0")")"
          export TILE_COMPILE_UI_DIR="$HERE/usr/share/tile_compile/ui"
          exec "$HERE/usr/bin/tile_compile_web_backend" "$@"
          EOF
          chmod +x AppDir/AppRun
          ./linuxdeploy-x86_64.AppImage \
            --appdir AppDir \
            --executable AppDir/usr/bin/tile_compile_web_backend \
            --output appimage
          mv TileCompile-x86_64.AppImage TileCompile-linux-x86_64.AppImage

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: linux-build
          path: |
            TileCompile-linux-x86_64.AppImage

  build-macos-arm:
    runs-on: macos-14     # Apple Silicon
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: brew install cmake ninja eigen opencv yaml-cpp curl
      - name: Build
        run: |
          cmake -B build -S tile_compile_cpp \
            -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_WEB_BACKEND=ON
          cmake --build build --target tile_compile_web_backend tile_compile_runner tile_compile_cli
      - name: Package .app
        run: |
          mkdir -p TileCompile.app/Contents/MacOS
          mkdir -p TileCompile.app/Contents/Resources/ui
          cp build/tile_compile_web_backend TileCompile.app/Contents/MacOS/
          cp build/tile_compile_runner      TileCompile.app/Contents/MacOS/
          cp build/tile_compile_cli         TileCompile.app/Contents/MacOS/
          cp -r web_frontend/dist/          TileCompile.app/Contents/Resources/ui/
          # Info.plist und AppLauncher-Script (öffnet Browser automatisch)
          zip -r TileCompile-macos-arm64.zip TileCompile.app
      - uses: actions/upload-artifact@v4
        with:
          name: macos-arm64
          path: TileCompile-macos-arm64.zip

  build-macos-x86:
    runs-on: macos-13     # Intel
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: brew install cmake ninja eigen opencv yaml-cpp curl
      - name: Build
        run: |
          cmake -B build -S tile_compile_cpp \
            -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_WEB_BACKEND=ON
          cmake --build build --target tile_compile_web_backend tile_compile_runner tile_compile_cli
      - uses: actions/upload-artifact@v4
        with:
          name: macos-x86_64
          path: build/tile_compile_web_backend

  build-windows:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - name: Install vcpkg dependencies
        run: |
          vcpkg install eigen3 opencv yaml-cpp curl --triplet x64-windows-static
      - name: Build
        run: |
          cmake -B build -S tile_compile_cpp `
            -DCMAKE_BUILD_TYPE=Release `
            -DBUILD_WEB_BACKEND=ON `
            -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
            -DVCPKG_TARGET_TRIPLET=x64-windows-static
          cmake --build build --config Release --target tile_compile_web_backend tile_compile_runner tile_compile_cli
      - uses: actions/upload-artifact@v4
        with:
          name: windows-x64
          path: |
            build/Release/tile_compile_web_backend.exe
            build/Release/tile_compile_runner.exe
            build/Release/tile_compile_cli.exe

  release:
    needs: [build-linux, build-macos-arm, build-macos-x86, build-windows]
    runs-on: ubuntu-22.04
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/download-artifact@v4
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            linux-build/TileCompile-linux-x86_64.AppImage
            macos-arm64/TileCompile-macos-arm64.zip
            macos-x86_64/tile_compile_web_backend
            windows-x64/tile_compile_web_backend.exe
            windows-x64/tile_compile_runner.exe
            windows-x64/tile_compile_cli.exe
```

---

## Teil 8: Linux-Kompatibilität (breite Distro-Abdeckung)

### Strategie: Alte glibc + statische Deps

Das größte Kompatibilitätsproblem auf Linux ist die **glibc-Version**.
Binaries die auf Ubuntu 22.04 (glibc 2.35) gebaut werden, laufen **nicht** auf
Ubuntu 20.04 (glibc 2.31) oder älteren Systemen.

**Lösung: Build auf Ubuntu 20.04 (`ubuntu-20.04` Runner) oder Verwendung eines alten Containers.**

```yaml
  build-linux:
    runs-on: ubuntu-20.04   # glibc 2.31 → breite Kompatibilität
```

Damit läuft die Binary auf:
- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- Fedora 34+
- openSUSE 15.3+
- Mint 20+
- Raspberry Pi OS (64-bit)

### AppImage als bevorzugtes Linux-Format

Das AppImage bündelt alle nötigen Shared Libraries (außer glibc und libfuse) via `linuxdeploy`.
Damit läuft es auf allen Linux-Distros mit glibc ≥ 2.31 **ohne Installation**:

```bash
chmod +x TileCompile-linux-x86_64.AppImage
./TileCompile-linux-x86_64.AppImage
# → öffnet Browser auf http://localhost:8000/ui/
```

### Statisch linkbare Bibliotheken

Für maximale Portabilität können folgende Libs statisch gelinkt werden:

| Bibliothek | Statisch linkbar | Hinweis |
|---|---|---|
| Crow / Asio | ✅ (header-only) | immer statisch |
| nlohmann/json | ✅ (header-only) | immer statisch |
| yaml-cpp | ✅ | `-DYAML_CPP_BUILD_SHARED_LIBS=OFF` |
| libcurl | ✅ | `CURL_STATICLIB`, zieht OpenSSL rein |
| OpenSSL | ✅ | für curl-Deps, mit `-static` |
| glibc | ❌ | niemals statisch linken (LGPL, Systemkompatibilität) |
| libstdc++ | ✅ (mit `-static-libstdc++`) | empfohlen für AppImage |

CMake-Flags für maximale Portabilität:

```cmake
# In web_backend_cpp/CMakeLists.txt
if (UNIX AND NOT APPLE)
    target_link_options(tile_compile_web_backend PRIVATE
        -static-libgcc
        -static-libstdc++)
endif()
```

---

## Abhängigkeiten-Übersicht

| Bibliothek | Zweck | Beschaffung |
|---|---|---|
| **Crow v1.2.0** | HTTP/WebSocket Server | FetchContent |
| **Asio (standalone)** | Async I/O für Crow | FetchContent |
| **nlohmann/json** | JSON | FetchContent oder System |
| **yaml-cpp** | YAML Parsing/Emitting | System (bereits vorhanden) |
| **libcurl** | HTTP-Downloads (ASTAP, Siril-Katalog) | System |
| **std::filesystem** | Pfad-Operationen | C++17 stdlib |
| **std::thread / std::mutex** | Parallelität | C++17 stdlib |

**Kein Python, kein pip, kein venv, kein numpy, kein matplotlib, kein yaml (Python).**
