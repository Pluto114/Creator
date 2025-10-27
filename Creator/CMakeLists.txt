cmake_minimum_required(VERSION 3.20)
project(Creator LANGUAGES CXX C)

# ========== 编译基础 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

if(MSVC)
    add_compile_options(/utf-8)
    # 与 vcpkg 默认一致，避免 LNK2038
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ========== 第三方源码/路径（仅 header-only 或本地源码） ==========
set(GLAD_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/glad/include")
set(STB_DIR          "${CMAKE_SOURCE_DIR}/external/stb")
set(NLOHMANN_DIR     "${CMAKE_SOURCE_DIR}/external/nlohmann/include")

# 你可以继续使用 <glm/...> 的标准包含；下文会额外加一个兼容路径让旧写法也能过
add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

# ========== ImGui 统一封装：vcpkg 优先，找不到回退源码 ==========
# 使用方式：add_imgui_target(IMGUI_TARGET) -> target_link_libraries(Creator PRIVATE ${IMGUI_TARGET})
function(add_imgui_target out_target)
    # 1) vcpkg / CONFIG
    find_package(imgui CONFIG QUIET)
    if (imgui_FOUND)
        add_library(thirdparty_imgui INTERFACE)
        target_link_libraries(thirdparty_imgui INTERFACE imgui::imgui)
        target_compile_definitions(thirdparty_imgui INTERFACE IMGUI_IMPL_OPENGL_LOADER_GLAD)
        set(${out_target} thirdparty_imgui PARENT_SCOPE)
        message(STATUS "[ImGui] Using vcpkg package (imgui::imgui)")
        return()
    endif()

    # 2) 回退到仓库源码：自动找包含 imgui.h 的目录
    set(_IMGUI_CANDIDATES
            "${CMAKE_SOURCE_DIR}/external/imgui"
            "${CMAKE_SOURCE_DIR}/external/imgui-master"
            "D:/Creator/Creator/external/imgui"
            "D:/Creator/Creator/external/imgui-master"
    )
    unset(IMGUI_DIR CACHE)
    foreach(_cand IN LISTS _IMGUI_CANDIDATES)
        if(EXISTS "${_cand}/imgui.h")
            set(IMGUI_DIR "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT IMGUI_DIR)
        message(FATAL_ERROR "[ImGui] Not found. Install via vcpkg (imgui[glfw-binding,opengl3-binding]) or place sources in external/imgui(.git).")
    endif()
    message(STATUS "[ImGui] Using vendored sources at: ${IMGUI_DIR}")

    set(_IMGUI_SRC
            ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_draw.cpp
            ${IMGUI_DIR}/imgui_tables.cpp
            ${IMGUI_DIR}/imgui_widgets.cpp
            ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
            ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    add_library(imgui_vendor STATIC ${_IMGUI_SRC})
    target_include_directories(imgui_vendor PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_compile_definitions(imgui_vendor PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

    # 这些头在编译 backends 时需要
    if(DEFINED GLAD_INCLUDE_DIR)
        target_include_directories(imgui_vendor PRIVATE ${GLAD_INCLUDE_DIR})
    endif()

    # backends 需要 GLFW 和 OpenGL 头；这里仅建立编译期依赖（包含路径）
    # 要求在调用此函数前已 find_package(glfw3) & find_package(OpenGL)
    if (TARGET glfw)
        target_link_libraries(imgui_vendor PRIVATE glfw)
    endif()
    if (TARGET OpenGL::GL)
        target_link_libraries(imgui_vendor PRIVATE OpenGL::GL)
    endif()

    add_library(thirdparty_imgui INTERFACE)
    target_link_libraries(thirdparty_imgui INTERFACE imgui_vendor)
    set(${out_target} thirdparty_imgui PARENT_SCOPE)
endfunction()

# ========== 源文件 ==========
file(GLOB_RECURSE APP_SOURCES
        "core/*.cpp"
        "graphics/*.cpp"
        "ui/*.cpp"
)
set(GLAD_SOURCES "${CMAKE_SOURCE_DIR}/external/glad/src/glad.c"
        core/Raycast.cpp
        core/Raycast.h)

add_executable(Creator
        main.cpp
        ${APP_SOURCES}
        ${GLAD_SOURCES}
        core/Raycast.cpp
        core/Raycast.h
)



# === tinyfiledialogs（C 源）===
# 源码在 ui/ 目录，头文件就是 tinyfiledialogs.h
add_library(tinyfiledialogs STATIC ${CMAKE_SOURCE_DIR}/ui/tinyfiledialogs.c)
target_include_directories(tinyfiledialogs PUBLIC ${CMAKE_SOURCE_DIR}/ui)

# Windows 上对话框需要这些系统库
if (WIN32)
    target_link_libraries(tinyfiledialogs PUBLIC comdlg32 ole32 uuid)
endif()

# 链接到主程序
target_link_libraries(Creator PRIVATE tinyfiledialogs)


# 头文件路径（ImGui 的 include/backends 由封装目标提供，这里无需再加）
target_include_directories(Creator PRIVATE
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/graphics
        ${CMAKE_SOURCE_DIR}/ui
        ${GLAD_INCLUDE_DIR}
        ${STB_DIR}
        ${NLOHMANN_DIR}
)

# ========== 依赖（均走 CONFIG/vcpkg） ==========
find_package(OpenCV CONFIG REQUIRED)
message(STATUS "成功找到 OpenCV: ${OpenCV_VERSION}")
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)


# ImGui 目标（统一别名）
add_imgui_target(IMGUI_TARGET)

# ========== 链接 ==========
target_link_libraries(Creator PRIVATE
        ${OpenCV_LIBS}       # OpenCV 传统变量
        OpenGL::GL
        glfw                 # 来自 glfw3 CONFIG 的目标名通常就是 glfw
        CURL::libcurl
        assimp::assimp
        glm::glm
         imguizmo::imguizmo
        ${IMGUI_TARGET}      # vcpkg 或 vendored
)

# Windows 额外系统库（按需）
if (WIN32)
    target_link_libraries(Creator PRIVATE dwmapi)
endif()

# ========== GLM 兼容“垫片”（让旧的 #include "gtc/..." 也能被找到） ==========
# 标准写法是 <glm/...>；以下代码把 <install>/include/glm 也加入 include 根
get_target_property(_GLM_INC_DIRS glm::glm INTERFACE_INCLUDE_DIRECTORIES)
set(_GLM_COMPAT_DIRS "")
foreach(_d ${_GLM_INC_DIRS})
    list(APPEND _GLM_COMPAT_DIRS "${_d}/glm")
endforeach()
target_include_directories(Creator PRIVATE ${_GLM_COMPAT_DIRS})
# 防止 Windows.h 定义 min/max 宏；顺便精简头
target_compile_definitions(Creator PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
# 需要 CMake >= 3.21 才有 $<TARGET_RUNTIME_DLLS:...> 和 COMMAND_EXPAND_LISTS
cmake_minimum_required(VERSION 3.21)

# 需要 CMake >= 3.21
cmake_minimum_required(VERSION 3.21)

if (WIN32)
    # 1) 从 imported targets 自动复制其运行时 DLL（vcpkg 出的）
    add_custom_command(TARGET Creator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "-- Deploying runtime DLLs from imported targets..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:Creator>
            $<TARGET_FILE_DIR:Creator>
            COMMAND ${CMAKE_COMMAND} -E echo "-- Done deploying imported target DLLs."
            COMMAND_EXPAND_LISTS
    )

    # 2) 找到 OpenCV world 对应的 bin 目录与版本号（4120 等）
    set(_OCV_BINDIR "")
    set(_OCV_VER "")
    set(_OCV_WORLD_DLL "")

    foreach(_lib ${OpenCV_LIBS})
        get_filename_component(_name "${_lib}" NAME_WE)          # e.g. opencv_world4120d
        if(_name MATCHES "^opencv_world([0-9]+)d?$")
            get_filename_component(_libdir "${_lib}" DIRECTORY)    # .../x64/vc16/lib
            get_filename_component(_vcdir  "${_libdir}/.." ABSOLUTE)
            set(_OCV_BINDIR "${_vcdir}/bin")                       # .../x64/vc16/bin
            # 去掉前缀和 Debug 后缀，得到纯版本号 4120
            string(REGEX REPLACE "^opencv_world" "" _ver "${_name}")
            string(REGEX REPLACE "d$" "" _ver "${_ver}")
            set(_OCV_VER "${_ver}")
            # world 对应的 DLL（按当前配置拷贝一次）
            set(_OCV_WORLD_DLL "${_OCV_BINDIR}/${_name}.dll")
            break()
        endif()
    endforeach()

    # 2.1 拷贝 world DLL
    if(_OCV_WORLD_DLL AND EXISTS "${_OCV_WORLD_DLL}")
        add_custom_command(TARGET Creator POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "-- Copy OpenCV world: ${_OCV_WORLD_DLL}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_OCV_WORLD_DLL}" $<TARGET_FILE_DIR:Creator>
        )
    endif()

    # 2.2 拷贝 videoio 插件（ffmpeg/gstreamer/intel_mfx）
    if(_OCV_BINDIR AND _OCV_VER)
        file(GLOB _OCV_VIDEO_PLUGINS
                "${_OCV_BINDIR}/opencv_videoio_ffmpeg${_OCV_VER}_64*.dll"
                "${_OCV_BINDIR}/opencv_videoio_gstreamer${_OCV_VER}_64*.dll"
                "${_OCV_BINDIR}/opencv_videoio_intel_mfx${_OCV_VER}_64*.dll"
        )
        if(_OCV_VIDEO_PLUGINS)
            add_custom_command(TARGET Creator POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E echo "-- Copy OpenCV videoio plugins:"
                    COMMAND ${CMAKE_COMMAND} -E echo "${_OCV_VIDEO_PLUGINS}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_OCV_VIDEO_PLUGINS} $<TARGET_FILE_DIR:Creator>
                    COMMAND_EXPAND_LISTS
            )
        else()
            message(WARNING "[Deploy] No OpenCV videoio plugins found under ${_OCV_BINDIR}. FFmpeg/GStreamer backends may be unavailable.")
        endif()

        # 2.3（可选）拷贝并行后端插件（无则忽略）
        file(GLOB _OCV_PARALLEL_PLUGINS
                "${_OCV_BINDIR}/opencv_core_parallel_*${_OCV_VER}_64*.dll"
        )
        if(_OCV_PARALLEL_PLUGINS)
            add_custom_command(TARGET Creator POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E echo "-- Copy OpenCV parallel plugins:"
                    COMMAND ${CMAKE_COMMAND} -E echo "${_OCV_PARALLEL_PLUGINS}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_OCV_PARALLEL_PLUGINS} $<TARGET_FILE_DIR:Creator>
                    COMMAND_EXPAND_LISTS
            )
        endif()
    endif()
endif()
