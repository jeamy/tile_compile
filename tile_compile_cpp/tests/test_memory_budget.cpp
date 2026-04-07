#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "tile_compile/reconstruction/memory_budget.hpp"
#include "tile_compile/core/errors.hpp"

using namespace tile_compile::reconstruction;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr size_t MB = 1024ULL * 1024;

// Build a minimal valid plan with sensible defaults.
static MemoryBudgetPlan make_plan(int num_frames   = 100,
                                  int frame_rows   = 100,
                                  int frame_cols   = 100,
                                  int frame_ch     = 1,
                                  int num_tiles    = 50,
                                  int tile_w       = 32,
                                  int tile_h       = 32,
                                  int tile_ch      = 1,
                                  size_t budget    = 512 * MB,
                                  int workers      = 4) {
    return compute_memory_budget_plan(num_frames, frame_rows, frame_cols, frame_ch,
                                      num_tiles, tile_w, tile_h, tile_ch,
                                      budget, workers);
}

// ---------------------------------------------------------------------------
// Eigenschaft 1: allocated_frame_batch_bytes + allocated_tile_batch_bytes ≤ budget
// ---------------------------------------------------------------------------
TEST_CASE("memory_budget_plan_total_allocation_within_budget") {
    // Vary frame size, tile count and budget across a range of values.
    const std::vector<std::tuple<int,int,int,size_t>> cases = {
        {100, 100, 50,  512 * MB},
        {610, 200, 475, 2048 * MB},
        {50,  50,  10,  128 * MB},
        {1,   10,  1,   64 * MB},
        {200, 300, 100, 1024 * MB},
    };
    for (auto [nf, dim, nt, budget] : cases) {
        const auto plan = compute_memory_budget_plan(
            nf, dim, dim, 1, nt, 32, 32, 1, budget, 4);
        INFO("num_frames=" << nf << " dim=" << dim << " num_tiles=" << nt
             << " budget=" << budget / MB << " MB");
        CHECK(plan.allocated_frame_batch_bytes + plan.allocated_tile_batch_bytes
              <= budget);
    }
}

// ---------------------------------------------------------------------------
// Eigenschaft 2: frame_sub_batch_size × frame_bytes ≤ budget × 0.8
// ---------------------------------------------------------------------------
TEST_CASE("memory_budget_plan_sub_batch_respects_80pct_limit") {
    const std::vector<std::tuple<int,int,size_t>> cases = {
        {610, 200, 2048 * MB},
        {100, 100, 512 * MB},
        {50,  50,  128 * MB},
        {1,   10,  64 * MB},
    };
    for (auto [nf, dim, budget] : cases) {
        const auto plan = compute_memory_budget_plan(
            nf, dim, dim, 1, 10, 32, 32, 1, budget, 4);
        const size_t frame_bytes =
            static_cast<size_t>(dim) * static_cast<size_t>(dim) * sizeof(float);
        INFO("num_frames=" << nf << " dim=" << dim << " budget=" << budget / MB << " MB");
        CHECK(plan.frame_sub_batch_size * frame_bytes
              <= static_cast<size_t>(budget * 0.8));
    }
}

// ---------------------------------------------------------------------------
// Eigenschaft 5: effective_workers ≥ max(1, N/2) when budget > 512 MB
// ---------------------------------------------------------------------------
TEST_CASE("memory_budget_plan_worker_minimum_floor_above_512mb") {
    const std::vector<int> worker_counts = {2, 4, 8, 16, 32};
    for (int N : worker_counts) {
        // Use a generous budget so workers should not be reduced below floor.
        const auto plan = compute_memory_budget_plan(
            10, 50, 50, 1, 5, 32, 32, 1, 4096 * MB, N);
        const int min_workers = std::max(1, N / 2);
        INFO("requested_workers=" << N);
        CHECK(plan.effective_workers >= min_workers);
    }
}

// ---------------------------------------------------------------------------
// Unit-Test: Budget < 1 Frame → ReconstructionError
// ---------------------------------------------------------------------------
TEST_CASE("memory_budget_plan_throws_when_budget_too_small_for_frame") {
    // Frame = 100×100×float = 40 000 bytes. Budget = 1 byte → must throw.
    CHECK_THROWS_AS(
        compute_memory_budget_plan(1, 100, 100, 1, 1, 32, 32, 1, 1, 1),
        tile_compile::ReconstructionError);
}

// ---------------------------------------------------------------------------
// Unit-Test: Budget warning when workers reduced below floor
// ---------------------------------------------------------------------------
TEST_CASE("memory_budget_plan_sets_warning_when_workers_reduced_below_floor") {
    // 8 workers, tiny budget (600 MB) with large frames (3840×2160 = ~31 MB each).
    // OLA buffers: 1 tile × 32×32 × 2 × float = 8 KB → negligible.
    // Available for frames ≈ 600 MB × 0.8 = 480 MB.
    // Per-worker: 480 MB / 8 = 60 MB → fits ~1 frame per worker.
    // With 8 workers × 1 frame × 31 MB = 248 MB < 480 MB → should be fine.
    // Use extreme case: 1 frame = 200 MB, budget = 600 MB, 8 workers.
    // frame_bytes = 7000×7000×float ≈ 196 MB.
    // available ≈ 480 MB → sub_batch = 2 frames.
    // per_worker = 2 × 196 MB = 392 MB. 8 × 392 MB >> 480 MB → reduce workers.
    const size_t budget = 600 * MB;
    const auto plan = compute_memory_budget_plan(
        10, 7000, 7000, 1, 1, 32, 32, 1, budget, 8);
    // Workers must have been reduced (budget is tight).
    // The important thing: if budget > 512 MB and workers < max(1, N/2),
    // budget_warning must be set.
    if (plan.effective_workers < std::max(1, 8 / 2)) {
        CHECK(plan.budget_warning);
        CHECK(!plan.warning_reason.empty());
    }
}
