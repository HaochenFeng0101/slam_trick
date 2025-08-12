# src/BUILD

cc_library(
    name = "sort_lib",  # Renamed to avoid confusion with the binary
    srcs = [
        "sort.cpp",
    ],
    deps = ["//include:sort_headers"],
    # hdrs = [
    #     "../include/sort.hpp",
    # ],
    # includes = ["../include"],  # Include current directory for headers
)

cc_binary(
    name = "sort_app",
    srcs = ["main.cpp"],
    deps = [":sort_lib"],  # Correctly depend on the library
)