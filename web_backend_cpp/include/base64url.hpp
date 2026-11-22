#pragma once
#include <string>

/// Decodes base64url (RFC 4648 §5, padding optional). Returns "" on any
/// character outside the alphabet.
inline std::string decode_base64url(std::string value) {
    for (char& ch : value) {
        if (ch == '-') ch = '+';
        else if (ch == '_') ch = '/';
    }
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int buffer = 0;
    int bits_collected = 0;
    for (char ch : value) {
        if (ch == '=') break;
        const auto pos = alphabet.find(ch);
        if (pos == std::string::npos) return "";
        buffer = ((buffer << 6) | static_cast<int>(pos)) & 0xFFFFFF;
        bits_collected += 6;
        if (bits_collected >= 8) {
            bits_collected -= 8;
            out.push_back(static_cast<char>((buffer >> bits_collected) & 0xFF));
        }
    }
    return out;
}

/// A run id as the frontend sends it in a path: "b64_<base64url>" is decoded,
/// anything else (or an undecodable b64_ value) comes back unchanged.
inline std::string decode_b64_run_id(const std::string& run_id) {
    if (run_id.rfind("b64_", 0) != 0) return run_id;
    const std::string decoded = decode_base64url(run_id.substr(4));
    return decoded.empty() ? run_id : decoded;
}
