#include "services/pi/pi_action_plan.hpp"
#include "services/pi/pi_action_validator.hpp"

#include "backend_test_harness.hpp"

int main() {
    try {
        const nlohmann::json updates = nlohmann::json::array({
            {
                {"id", "set_color_mode"},
                {"path", "data.color_mode"},
                {"value", "MONO"},
                {"reason", "mono frames detected"},
                {"confidence", 0.9}
            },
            {
                {"id", "bad_type"},
                {"path", "data.color_mode"},
                {"value", 12}
            },
            {
                {"id", "bad_enum"},
                {"path", "data.color_mode"},
                {"value", "RGB"}
            },
            {
                {"id", "unknown_path"},
                {"path", "data.missing"},
                {"value", true}
            }
        });
        const nlohmann::json analysis = {
            {"schema_version", "pi.scan-analysis.v1"},
            {"summary", "fixture summary"},
            {"confidence", 0.8},
            {"warnings", nlohmann::json::array()}
        };
        const nlohmann::json schema_by_path = {
            {"data.color_mode", {
                {"type", "string"},
                {"enum", {"OSC", "MONO"}}
            }}
        };

        const auto plan = tile_compile::pi::build_scan_analysis_action_plan(analysis, updates);
        expect_equal(plan["schema_version"].get<std::string>(), "pi.action-plan.v1", "action plan schema");
        expect_equal(plan["source_schema_version"].get<std::string>(), "pi.scan-analysis.v1", "action plan source schema");
        expect_equal(static_cast<long>(plan["actions"].size()), 4L, "action plan action count");
        expect_equal(plan["actions"][0]["type"].get<std::string>(), "config.set", "action plan action type");
        expect_equal(plan["actions"][0]["path"].get<std::string>(), "data.color_mode", "action plan action path");

        const auto plan_validation = tile_compile::pi::validate_action_plan_shape(plan);
        expect_true(plan_validation["valid"].get<bool>(), "valid generated action plan");

        auto invalid_plan = plan;
        invalid_plan["actions"][0]["id"] = invalid_plan["actions"][1]["id"];
        const auto invalid_validation = tile_compile::pi::validate_action_plan_shape(invalid_plan);
        expect_true(!invalid_validation["valid"].get<bool>(), "duplicate action id rejected");

        const auto prevalidation = tile_compile::pi::prevalidate_config_updates(updates, schema_by_path);
        expect_equal(static_cast<long>(prevalidation["validated_updates"].size()), 1L, "prevalidated update count");
        expect_equal(static_cast<long>(prevalidation["rejected_updates"].size()), 3L, "prevalidated rejected count");
        expect_equal(prevalidation["rejected_updates"][0]["reject_reason"].get<std::string>(), "wrong_type",
                     "wrong type reject reason");
        expect_equal(prevalidation["rejected_updates"][1]["reject_reason"].get<std::string>(), "enum_mismatch",
                     "enum mismatch reject reason");
        expect_equal(prevalidation["rejected_updates"][2]["reject_reason"].get<std::string>(), "unknown_path",
                     "unknown path reject reason");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
