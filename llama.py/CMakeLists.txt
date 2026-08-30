find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_infrastructure MODULE
    llama/cc/module.cc)

target_compile_features(_infrastructure PRIVATE cxx_std_17)
target_link_libraries(_infrastructure PRIVATE llama)

set(INFRASTRUCTURE_PYTHON_DIR "${CMAKE_BINARY_DIR}/python/llama")
set_target_properties(_infrastructure PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${INFRASTRUCTURE_PYTHON_DIR}")

add_custom_command(TARGET _infrastructure POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${INFRASTRUCTURE_PYTHON_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/llama/__init__.py"
            "${INFRASTRUCTURE_PYTHON_DIR}/__init__.py"
    COMMENT "Preparing Infrastructure Python package")
