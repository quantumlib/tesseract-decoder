# external/boost.BUILD
package(default_visibility = ["//visibility:public"])

# A cc_library for the Boost headers themselves.
cc_library(
    name = "boost_headers",
    hdrs = glob(["boost/**/*.hpp"]),
    includes = ["."],
)

# A specific target for dynamic_bitset, which is header-only
# and depends on the main headers.
cc_library(
    name = "dynamic_bitset",
    deps = [":boost_headers"],
)
