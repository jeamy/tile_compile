# Tile Compile – Parallel Execution with RabbitMQ

**Status:** Optional extension document
**Scope:** compatible with *tile_basierte_qualitatsrekonstruktion_methodik.md*, PROC (Clean Break) and YAML

---

## 1. Goal and context

This document describes an **optional, non‑normative parallelization architecture** for *Tile Compile* based on **RabbitMQ**. The goal is **horizontal scaling** of the compute‑intensive tile analysis across multiple worker processes or hosts without violating the deterministic semantics of the methodology.

Parallelization is:

- **optional** (single‑node execution remains the reference)
- **deterministic**
- **fault‑tolerant**
- **reproducible**

---

## 2. Design principles

The parallelization follows five hard principles:

1. **No change to the methodology**
   Parallelization affects execution only, not the mathematical definition.

2. **Master‑driven reduction**
   All non‑associative operations (overlap‑add, summation) happen centrally.

3. **Idempotent tasks**
   Each task can be re‑executed without side effects.

4. **Deterministic ordering**
   Result aggregation happens in a stable, deterministic order.

5. **Explicit separation of production and diagnostics**
   Diagnostic artifacts must not block the production path.

---

## 3. Architecture overview

### 3.1 Roles

- **Master**
  - reads configuration
  - creates tile tasks
  - aggregates results
  - performs overlap‑add

- **Worker**
  - consumes tile tasks
  - runs the full tile analysis and tile reconstruction
  - returns reconstructed tile blocks

- **RabbitMQ**
  - task distribution
  - retry and failure handling

### 3.2 Operating modes & orchestration

This section describes the **organizational distribution** (main server vs workers) across two operating modes.

#### Mode A: Local (single host)

Goal: fast development/debugging of the parallelization with minimal infrastructure.

- Master and workers run on the same host.
- RabbitMQ runs locally (e.g., Docker Compose service).
- Workers scale horizontally via multiple local processes/containers.
- Diagnostic queues/artifacts can be enabled but must not block the production path.

Recommended verification (automatable):

- RabbitMQ reachable (ping)
- queues exist: `tile.compute`, `tile.results`, `tile.diagnostics`, `tile.dlq`
- workers consume tasks (consumer count > 0)
- dedupe works: for (`run_id`, `tile_id`) exactly one accepted result exists

#### Mode B: Production (main server + remote workers)

Goal: horizontal scaling across multiple hosts.

- **Main server** runs the central RabbitMQ instance.
- **Worker hosts** connect to the main server via an overlay network (e.g., Tailscale/NetBird).
- RabbitMQ should ideally only be reachable via the overlay (not publicly exposed).
- Workers run as independent deployments (e.g., Docker Compose on worker hosts).

Recommended verification (automatable):

- main server: RabbitMQ ping
- worker hosts reachable (overlay ping / SSH optional)
- each worker can:
  - consume from task queue
  - publish to results queue
  - access shared/object storage for `tile_data_ref`
- end‑to‑end: tasks are processed, results appear in `tile.results`, and the master can aggregate all tiles

Note:

- Production mode does not change semantics. Determinism is still enforced via master aggregation.

### 3.3 Run lifecycle (method‑level)

This describes a run in a way that can be represented by a setup script (local/production).

#### Phase 0: Run initialization

- master generates `run_id` (UUID) and writes a **frames manifest** (sorted frame list)
- master computes `frames_manifest_id` (hash of the manifest)
- master computes `config_hash` (hash of the relevant parts of `tile_compile.yaml`)
- master generates the tile grid deterministically

#### Phase 1: Dispatch

- master publishes tile tasks to `tile.compute`
- each task contains at least: `run_id`, `correlation_id`, `frames_manifest_id`, `config_hash`, `tile_id`, `tile_bbox`

#### Phase 2: Compute (worker)

- worker validates task compatibility:
  - `frames_manifest_id` known/retrievable
  - `config_hash` matches the loaded configuration
  - `input_stage` is satisfied (e.g., `registered_normalized`)
- worker processes the tile deterministically and writes the result to `tile_data_ref`
- worker publishes a result to `tile.results` and ACKs the task only afterwards

#### Phase 3: Aggregate (master)

- master collects results until all tiles are complete
- master deduplicates by (`run_id`, `tile_id`) and validates `tile_data_checksum`
- master performs deterministic overlap‑add (stable tile ordering)

#### Phase 4: Finish & cleanup

- diagnostics are collected optionally
- `tile.compute`/`tile.results` can be isolated per run via TTL/auto‑delete or routing keys

### 3.4 Worker inventory & registration (production)

In a setup‑script pattern, the worker side is often described via an inventory list/env configuration.

Minimum recommended fields (conceptual):

- worker ID (name)
- overlay address (Tailscale/NetBird IP or DNS)
- worker capacity (e.g., number of parallel consumers/processes)

Semantics:

- workers are **stateless** with respect to global run aggregation
- workers may join/leave dynamically (RabbitMQ consumer model)
- the master may optionally verify whether expected workers are online

### 3.5 Overlay network (Tailscale/NetBird)

For remote workers, an overlay network is recommended so RabbitMQ does not need to be publicly exposed.

Recommendations:

- RabbitMQ binds to the overlay interface (or firewall allows overlay only)
- management UI (if enabled) also overlay only
- workers connect only via the overlay address

Note:

- RabbitMQ permissions (vhost/users) still apply; overlay does not automatically mean “trusted”.

### 3.6 Storage variants for `tile_data_ref`

Parallelization only works if the master can access the tile data produced by workers.

Options:

- shared FS (NFS/SMB) in the overlay
  - pro: simple (path as `tile_data_ref`)
  - con: WAN performance/fragility
- object storage (S3/MinIO)
  - pro: WAN‑friendly, scalable
  - con: credentials/policies required

Rule of thumb:

- RabbitMQ for the control plane (JSON), storage for the data plane (tile blocks).

---

## 4. Task granularity

### 4.1 Standard: tile tasks (recommended)

**One task corresponds to one tile** *t* across **all frames** *f*.

```text
Task = Tile t × Frames [0…N]
```

**Pros:**

- maximum parallelism
- minimal dependencies
- cache‑friendly I/O
- ideal for CPU‑bound workloads

---

### 4.2 Option: tile batch tasks (supertiles)

Multiple tiles are grouped into one task.

**Useful when:**

- tiles are very small
- RabbitMQ overhead is high

**Trade‑off:** less parallelism, better I/O amortization.

---

### 4.3 Option: frame chunks within one tile (experimental)

Split one tile into multiple tasks, each processing only a range of frames.

**Only useful when:**

- extremely many frames
- very fast local I/O (NVMe, RAM disk)

**Cons:**

- complex partial reduction
- higher numerical sensitivity

---

## 5. Task payload (minimal definition)

```json
{
  "task_type": "tile_compute",
  "correlation_id": "uuid",
  "run_id": "uuid",
  "config_hash": "sha256",
  "frames_manifest_id": "sha256",
  "input_stage": "registered_normalized",
  "tile_id": 123,
  "tile_bbox": [x0, y0, w, h],
  "frame_index_range": [0, 999],
  "metrics_config": { ... },
  "seed": 42
}
```

### Required fields

- `tile_id` – unique tile identifier
- `tile_bbox` – spatial bounds
- `frame_index_range` – deterministic frame range
- `seed` – deterministic initialization

Notes:

- `frames_manifest_id` references a deterministic frame list (sorted paths + optional checksums).
- `config_hash` ensures workers use exactly the same configuration.
- `input_stage` specifies whether workers consume already normalized frames.

---

## 6. Result payload

### 6.1 Production data

```json
{
  "correlation_id": "uuid",
  "run_id": "uuid",
  "tile_id": 123,
  "tile_data_ref": "path-or-object-key",
  "tile_data_checksum": "sha256",
  "tile_data_dtype": "float32",
  "tile_data_shape": [h, w],
  "sum_weights": "ΣW",
  "tile_median": "after bg subtraction"
}
```

Note:

- Large binary data should **not** be transported in RabbitMQ. Use `tile_data_ref` as a reference to a filesystem/object storage.

### 6.2 Diagnostics (separate queue)

- Q_local histograms
- tile weight distributions
- QA maps

These artifacts must be **non‑blocking**.

---

## 7. Queues and routing

Recommended queues:

- `tile.compute` – production tasks
- `tile.results` – tile results
- `tile.diagnostics` – diagnostics
- `tile.dlq` – dead‑letter queue

RabbitMQ features:

- prefetch limit (e.g., 1–2)
- priority queues (optional)
- manual ACKs

Message size note:

- `tile.compute` and `tile.results` contain only small JSON payloads.
- tile data are stored outside RabbitMQ via `tile_data_ref`.

---

## 8. End‑to‑end flow

### 8.1 Master

1. reads YAML
2. generates tile grid
3. creates deterministic tasks
4. publishes tasks to `tile.compute`

---

### 8.2 Worker

1. consumes task
2. loads required frames (locally cached)
3. computes local metrics and weights
4. reconstructs tile
5. publishes result
6. ACK

---

### 8.3 Aggregation

1. master collects all tile results
2. deduplicates results by (`run_id`, `tile_id`) and validates `tile_data_checksum`
3. sorts by `tile_id`
4. performs deterministic overlap‑add
5. writes synthetic frames

---

## 9. Determinism and reproducibility

### Rules

- no floating‑point reduction in random order
- aggregation only in the master
- stable sorting by `tile_id`

### Seed recommendation

```text
seed = uint32(sha256(run_id + ":" + tile_id)[0:4])
```

Note:

- do not use language/process‑dependent hash functions.

---

## 10. Fault tolerance

### Idempotency

- tasks have no global side effects
- results are written atomically

Practically:

- result key = (`run_id`, `tile_id`) is unique
- workers may rewrite an output as long as it atomically replaces the previous version

### Retries

- limited retry count
- exponential backoff

### Dead‑letter queue

- failed tiles are isolated
- a run fails only after exceeding a threshold

---

## 11. Performance notes

- run workers as data‑local as possible
- cache frames in RAM or on NVMe
- avoid random file access

Note:

- in multi‑host setups, you need shared storage (NFS/object storage) so the master can read `tile_data_ref`.

---

## 12. Security / operations (minimal requirements)

- enable RabbitMQ authentication (user/pass or certificates)
- use TLS if workers are not on the same host
- isolate runs via routing‑key prefix per `run_id` or separate vhosts
- apply TTL/max‑length policies for diagnostic queues (diagnostics must not clog production)

## 13. Scope / non‑goals

This parallelization:

- does **not** replace the methodology
- does **not** change any mathematical definitions
- is **optional and configuration‑dependent**

Single‑node execution remains the reference for validation.

---

## 14. Summary

RabbitMQ‑based parallelization enables scalable, fault‑tolerant, deterministic tile analysis. It is modular, optional, and fully compatible with Methodology v2.
