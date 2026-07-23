#include "services/pi/pi_live_image_session.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <atomic>

static int failures = 0;

#define EXPECT(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
         else { std::cout << "ok: " << msg << "\n"; } } while(0)

static cv::Mat make_test_image(int w = 64, int h = 64) {
    cv::Mat img(h, w, CV_32FC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(0.25 + x * 0.01),
                static_cast<float>(0.25 + y * 0.01),
                static_cast<float>(0.25 + (x + y) * 0.005));
    return img;
}

static bool images_equal(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::sum(diff)[0] == 0;
}

int main() {
    try {
        tile_compile::pi::LiveImageSessionStore store;
        auto img = make_test_image();

        // --- create ---
        std::string sid = store.create("test_run", img);
        EXPECT(!sid.empty(), "create returns non-empty session_id");

        // Verify session exists and current == original
        bool found = store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
            EXPECT(s.run_id == "test_run", "session has correct run_id");
            EXPECT(images_equal(s.current_fits, s.original_fits),
                   "current_fits == original_fits after create");
        });
        EXPECT(found, "with_session finds created session");

        // --- apply_operation (invertible) ---
        {
            nlohmann::json op = {
                {"type", "brightness"},
                {"params", {{"midtones", 0.1}, {"shadows", 0.0}, {"highlights", 0.0}}}
            };
            auto res = store.apply_operation(sid, op);
            EXPECT(res.success, "apply_operation(brightness) succeeds");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.size() == 1, "undo_stack has 1 entry");
                EXPECT(s.redo_stack.empty(), "redo_stack is empty");
                EXPECT(!images_equal(s.current_fits, s.original_fits),
                       "current_fits changed after operation");
            });
        }

        // --- undo ---
        {
            auto res = store.undo(sid);
            EXPECT(!res.image.empty(), "undo returns image");
            EXPECT(res.can_redo, "can_redo is true after undo");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.empty(), "undo_stack empty after undo");
                EXPECT(s.redo_stack.size() == 1, "redo_stack has 1 entry");
                EXPECT(s.operation_history.empty(), "operation_history mirrors current state after undo");
                // Undo rebuilds from original, so exact equality is expected.
                cv::Mat diff;
                cv::absdiff(s.current_fits, s.original_fits, diff);
                double max_diff = 0;
                cv::minMaxLoc(diff, nullptr, &max_diff);
                EXPECT(max_diff < 1e-6,
                       "current_fits exactly restored after undo");
            });
        }

        // --- redo ---
        {
            auto res = store.redo(sid);
            EXPECT(!res.image.empty(), "redo returns image");
            EXPECT(res.can_undo, "can_undo is true after redo");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.size() == 1, "undo_stack has 1 entry after redo");
                EXPECT(s.redo_stack.empty(), "redo_stack empty after redo");
                EXPECT(s.operation_history.size() == 1, "operation_history restored after redo");
                EXPECT(!images_equal(s.current_fits, s.original_fits),
                       "current_fits changed after redo");
            });
        }

        // --- reset ---
        {
            auto img_reset = store.reset(sid);
            EXPECT(!img_reset.empty(), "reset returns image");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(images_equal(s.current_fits, s.original_fits),
                       "current_fits == original after reset");
                EXPECT(s.undo_stack.empty(), "undo_stack empty after reset");
                EXPECT(s.redo_stack.empty(), "redo_stack empty after reset");
                EXPECT(s.adjust_count == 0, "adjust_count == 0 after reset");
            });
        }

        // --- adjust ---
        {
            nlohmann::json step = {
                {"type", "contrast"},
                {"params", {{"amount", 0.1}}}
            };
            store.set_adjust_step(sid, step);

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.adjust_count == 0, "adjust_count == 0 after set_adjust_step");
            });

            auto res_inc = store.apply_adjust(sid, "increase");
            EXPECT(res_inc.success, "apply_adjust(increase) succeeds");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.adjust_count == 1, "adjust_count == 1 after increase");
            });

            auto res_dec = store.apply_adjust(sid, "decrease");
            EXPECT(res_dec.success, "apply_adjust(decrease) succeeds");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.adjust_count == 0, "adjust_count == 0 after decrease");
            });

            store.reset(sid);
            nlohmann::json sharpen_step = {
                {"type", "sharpen"},
                {"params", {{"amount", 0.1}, {"radius", 2.0}}}
            };
            store.set_adjust_step(sid, sharpen_step);
            auto sharp_inc = store.apply_adjust(sid, "increase");
            EXPECT(sharp_inc.success, "sharpen +/- increase succeeds");
            auto sharp_dec = store.apply_adjust(sid, "decrease");
            EXPECT(sharp_dec.success && sharp_dec.image.size() == img.size(),
                   "sharpen +/- decrease rebuilds successfully");
        }

        // --- non-invertible ops use exact pre-operation snapshots ---
        {
            store.reset(sid);
            nlohmann::json op = {
                {"type", "clahe"},
                {"params", {{"cliplimit", 3.0}, {"tilesize", 8}}}
            };
            auto res = store.apply_operation(sid, op);
            EXPECT(res.success, "apply_operation(clahe) succeeds");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.size() == 1, "undo_stack has 1 entry after clahe");
            });

            // Undo must restore the exact pre-operation pixels.
            auto undo_res = store.undo(sid);
            EXPECT(!undo_res.image.empty(), "undo after clahe returns image");

            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(images_equal(s.current_fits, s.original_fits),
                       "current_fits restored after clahe undo snapshot");
            });
            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(!s.edit_history.empty() &&
                           s.edit_history.back().value("action", "") == "undo",
                       "edit_history records undo action");
            });

            nlohmann::json threshold = {
                {"type", "threshold"},
                {"params", {{"black_point", 0.2}, {"white_point", 0.8}}}
            };
            auto threshold_res = store.apply_operation(sid, threshold);
            EXPECT(threshold_res.success, "apply_operation(threshold) succeeds");
            auto threshold_undo = store.undo(sid);
            EXPECT(images_equal(threshold_undo.image, img),
                   "threshold undo restores exact snapshot");
            auto threshold_redo = store.redo(sid);
            EXPECT(threshold_redo.image.size() == img.size(),
                   "threshold redo restores post-operation snapshot");
            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.edit_history.back().value("action", "") == "redo",
                       "edit_history records redo action");
            });
            nlohmann::json reset_op = {{"type", "reset"}, {"params", nlohmann::json::object()}};
            auto reset_result = store.apply_operation(sid, reset_op);
            EXPECT(reset_result.success, "reset operation succeeds in session store");
            store.with_session(sid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.empty() && s.operation_history.empty(),
                       "reset operation clears current edit history");
                EXPECT(s.chat_history.empty(), "reset operation clears chat history");
            });

            nlohmann::json repeat_op = {
                {"type", "sharpen"}, {"params", {{"amount", 0.1}, {"radius", 2.0}}}
            };
            auto first_repeat = store.apply_operation(sid, repeat_op);
            EXPECT(first_repeat.success, "repeat source operation succeeds");
            auto repeated = store.repeat_operation(sid);
            EXPECT(repeated.success, "repeat operation reapplies identical params");
        }

        // --- chat history ---
        {
            store.reset(sid);
            store.append_chat(sid, "user", "helle das Bild auf");
            store.append_chat(sid, "assistant", "Mitteltone angehoben.",
                              nlohmann::json::array({{{"type", "brightness"}, {"params", {{"midtones", 0.1}}}}}));

            auto history = store.get_chat_history(sid);
            EXPECT(history.is_array() && history.size() == 2,
                   "chat_history has 2 entries");
            EXPECT(history[0]["role"] == "user", "first chat entry is user");
            EXPECT(history[1]["role"] == "assistant", "second chat entry is assistant");
            EXPECT(history[1].contains("operations"), "assistant entry has operations");
        }

        // --- operation history ---
        {
            store.reset(sid);
            nlohmann::json op = {{"type", "contrast"}, {"params", {{"amount", 0.2}}}};
            store.apply_operation(sid, op);

            auto hist = store.get_operation_history(sid);
            EXPECT(hist.is_array() && hist.size() == 1,
                   "operation_history has 1 entry");
            EXPECT(hist[0]["type"] == "contrast", "operation_history entry type");
            EXPECT(hist[0]["source"] == "chat", "operation_history entry source");
        }

        // --- evict_expired ---
        {
            // Create a session and manually age it
            std::string old_sid = store.create("old_run", img);
            store.with_session(old_sid, [&](tile_compile::pi::LiveImageSession& s) {
                s.last_accessed = std::chrono::steady_clock::now() - std::chrono::seconds(3600);
            });

            store.evict_expired(1800, 5);

            bool old_found = store.with_session(old_sid, [](tile_compile::pi::LiveImageSession&) {});
            EXPECT(!old_found, "expired session is evicted");

            bool active_found = store.with_session(sid, [](tile_compile::pi::LiveImageSession&) {});
            EXPECT(active_found, "active session still exists after evict");
        }

        // --- max sessions LRU ---
        {
            tile_compile::pi::LiveImageSessionStore store2;
            std::vector<std::string> ids;
            for (int i = 0; i < 7; ++i) {
                ids.push_back(store2.create("run" + std::to_string(i), img));
            }
            store2.evict_expired(1800, 5);

            int count = 0;
            for (const auto& id : ids) {
                if (store2.with_session(id, [](tile_compile::pi::LiveImageSession&) {}))
                    ++count;
            }
            EXPECT(count == 5, "only 5 sessions survive LRU eviction");
        }

        // --- close ---
        {
            std::string close_sid = store.create("close_run", img);
            store.close(close_sid);
            bool found = store.with_session(close_sid, [](tile_compile::pi::LiveImageSession&) {});
            EXPECT(!found, "closed session is removed");
        }

        // --- thread safety ---
        {
            tile_compile::pi::LiveImageSessionStore store3;
            std::string tsid = store3.create("thread_run", make_test_image(32, 32));

            nlohmann::json op = {{"type", "brightness"},
                                 {"params", {{"midtones", 0.05}, {"shadows", 0.0}, {"highlights", 0.0}}}};

            std::atomic<int> success_count{0};
            std::vector<std::thread> threads;
            for (int t = 0; t < 2; ++t) {
                threads.emplace_back([&]() {
                    for (int i = 0; i < 50; ++i) {
                        auto res = store3.apply_operation(tsid, op);
                        if (res.success) success_count++;
                    }
                });
            }
            for (auto& th : threads) th.join();

            EXPECT(success_count == 100, "all 100 threaded operations succeeded");
            store3.with_session(tsid, [&](tile_compile::pi::LiveImageSession& s) {
                EXPECT(s.undo_stack.size() == 100, "undo_stack has 100 entries after threading test");
            });
        }

    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        ++failures;
    }

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "\nAll tests passed\n";
    return 0;
}
