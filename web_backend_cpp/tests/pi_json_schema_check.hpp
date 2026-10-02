#pragma once
// Minimal JSON-Schema (draft-07 subset) checker shared by the PI decision tests: enough to prove
// that builder/resolver output and the committed schemas agree (required / additionalProperties /
// $ref / type / enum / const / not / allOf-if-then / properties / items). Returns an empty string
// when valid, otherwise a path-qualified reason.
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pi_test {
using nlohmann::json;
// ---- minimal JSON-Schema (draft-07 subset) checker: enough to prove the builder's output and
// the committed pi.decision-state.v1 schema agree (required / additionalProperties / $ref /
// type / enum / const / properties / items). Returns an empty string when valid. ----
inline std::string schema_check(const json& schema, const json& inst, const json& root, const std::string& path) {
    if (schema.contains("$ref")) {
        const std::string ref = schema["$ref"].get<std::string>();
        json node = root;
        size_t pos = 2;  // skip "#/"
        while (pos <= ref.size()) {
            const size_t next = ref.find('/', pos);
            node = node.at(ref.substr(pos, next == std::string::npos ? std::string::npos : next - pos));
            if (next == std::string::npos) break;
            pos = next + 1;
        }
        return schema_check(node, inst, root, path);
    }
    if (schema.contains("not")) return schema_check(schema["not"], inst, root, path).empty() ? path + ": matched 'not'" : "";
    if (schema.contains("const") && inst != schema["const"]) return path + ": const mismatch";
    if (schema.contains("enum")) {
        bool hit = false;
        for (const auto& e : schema["enum"]) hit = hit || e == inst;
        if (!hit) return path + ": not in enum (" + inst.dump() + ")";
    }
    if (schema.contains("type")) {
        std::vector<std::string> types;
        if (schema["type"].is_string()) types.push_back(schema["type"]);
        else for (const auto& t : schema["type"]) types.push_back(t);
        bool ok = false;
        for (const auto& t : types) {
            ok = ok || (t == "object" && inst.is_object()) || (t == "array" && inst.is_array()) ||
                 (t == "string" && inst.is_string()) || (t == "null" && inst.is_null()) ||
                 (t == "boolean" && inst.is_boolean()) || (t == "integer" && inst.is_number_integer()) ||
                 (t == "number" && inst.is_number());
        }
        if (!ok) return path + ": wrong type " + inst.dump().substr(0, 60);
    }
    if (schema.contains("allOf"))
        for (const auto& sub : schema["allOf"]) {
            if (sub.contains("if")) {
                if (schema_check(sub["if"], inst, root, path).empty() && sub.contains("then")) {
                    const auto e = schema_check(sub["then"], inst, root, path);
                    if (!e.empty()) return e;
                }
            }
        }
    if (inst.is_object()) {
        if (schema.contains("required"))
            for (const auto& k : schema["required"])
                if (!inst.contains(k.get<std::string>())) return path + ": missing required '" + k.get<std::string>() + "'";
        const json props = schema.value("properties", json::object());
        for (auto it = inst.begin(); it != inst.end(); ++it) {
            if (props.contains(it.key())) {
                const auto e = schema_check(props[it.key()], it.value(), root, path + "." + it.key());
                if (!e.empty()) return e;
            } else if (schema.contains("additionalProperties")) {
                const json& ap = schema["additionalProperties"];
                if (ap.is_boolean() && !ap.get<bool>()) return path + ": unexpected property '" + it.key() + "'";
                if (ap.is_object()) {
                    const auto e = schema_check(ap, it.value(), root, path + "." + it.key());
                    if (!e.empty()) return e;
                }
            }
        }
    }
    if (inst.is_array() && schema.contains("items"))
        for (size_t i = 0; i < inst.size(); ++i) {
            const auto e = schema_check(schema["items"], inst[i], root, path + "[" + std::to_string(i) + "]");
            if (!e.empty()) return e;
        }
    return "";
}


} // namespace pi_test
