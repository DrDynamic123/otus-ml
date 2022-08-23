cmake_minimum_required(VERSION 3.23)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VESRION 0.0.${PATCH_VERSION})

project(fashio_mnist VERSION ${PROJECT_VESRION})

include_directories("inc/")

aux_source_directory(src SOURCES)

add_executable(fashio_mnist ${SOURCES})

set_target_properties(fashio_mnist PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(fashio_mnist
    PRIVATE "${CMAKE_BINARY_DIR}"
)

include(CTest)

if(BUILD_TESTING)

    include(FetchContent)
    FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
    )
    if (WIN32)
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif (WIN32)

    FetchContent_MakeAvailable(googletest)

    add_executable(test_fashio_mnist tests/test_logreg_classifier.cpp
                                     src/helpers.cpp
                                     src/logreg_classifier.cpp
                                     src/fashion_classifier.cpp)

    set_target_properties(test_fashio_mnist PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    target_link_libraries(test_fashio_mnist
        gtest_main
    )
    
    include(GoogleTest)
endif()

if (MSVC)
    target_compile_options(fashio_mnist PRIVATE
        /W4
    )
    if(BUILD_TESTING)
        target_compile_options(test_fashio_mnist PRIVATE
            /W4
        )
    endif()
else ()
    target_compile_options(fashio_mnist PRIVATE
        -Wall -Wextra -pedantic -Werror -static -pthread
    )
    if(BUILD_TESTING)
        target_compile_options(test_fashio_mnist PRIVATE
            -Wall -Wextra -pedantic -Werror -static -pthread
        )
    endif()
endif()

install(TARGETS fashio_mnist RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT example@example.com)
include(CPack)

if(BUILD_TESTING)
    add_test(test_fashio_mnist test_fashio_mnist)
endif()
