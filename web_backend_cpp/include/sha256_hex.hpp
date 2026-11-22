#pragma once
#include <openssl/evp.h>

#include <stdexcept>
#include <string>

/// Lowercase hex SHA-256 of the given bytes.
inline std::string sha256_hex(const std::string& bytes) {
    unsigned char output[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), output, &size, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA-256 digest failed");
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size * 2);
    for (unsigned int i = 0; i < size; ++i) {
        result += digits[output[i] >> 4];
        result += digits[output[i] & 15];
    }
    return result;
}
