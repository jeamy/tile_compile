 #if __has_include(<catch2/catch_test_macros.hpp>)
 #include <catch2/catch_test_macros.hpp>

 TEST_CASE("tests_bootstrap") {
     REQUIRE(true);
 }
 #else
 int tile_compile_tests_bootstrap() { return 0; }
 #endif
