cmake_minimum_required(VERSION 2.8)
project(darknet C CXX)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

function(detect_cuDNN)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

    find_path(CUDNN_INCLUDE cudnn.h
        PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
        DOC "Path to cuDNN include directory." )

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES libcudnn.so # libcudnn_static.a
        PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist}
        DOC "Path to cuDNN library.")

    if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
        set(HAVE_CUDNN  TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)

        file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

        # cuDNN v3 and beyond
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
            CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
            CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
            CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
            CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
            CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
            CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "???")
        else()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif()

        message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

        string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
        if(cuDNNVersionIncompatible)
            message(FATAL_ERROR "cuDNN version >3 is required.")
        endif()

        set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
        mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

    endif()
endfunction()

# hack for cmake external project

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})

add_definitions(-DDEBUG=0)
#add_definitions(-DDEBUG=1)
add_definitions(-DOPENCV=1)

# -- OpenCV --

find_package(OpenCV REQUIRED)

file(GLOB SOURCE_CPU_FILES "*.h" "*.c" "*.cpp")

list(REMOVE_ITEM SOURCE_CPU_FILES "${CMAKE_SOURCE_DIR}/cpu_gemm.c")
list(REMOVE_ITEM SOURCE_CPU_FILES "${CMAKE_SOURCE_DIR}/server.c")
#list(REMOVE_ITEM SOURCE_CPU_FILES "${CMAKE_SOURCE_DIR}/server.h")

list(APPEND SOURCE_CPU_FILES "${CMAKE_SOURCE_DIR}/detector_wrapper.h")
list(APPEND SOURCE_CPU_FILES "${CMAKE_SOURCE_DIR}/detector_wrapper.cpp")

list(APPEND CMAKE_C_FLAGS "-Wall -Wfatal-errors")

# CPU version

add_library(${PROJECT_NAME}_cpu SHARED ${SOURCE_CPU_FILES})
target_link_libraries(${PROJECT_NAME}_cpu pthread m ${OpenCV_LIBS})
install(TARGETS ${PROJECT_NAME}_cpu DESTINATION .)

find_package(CUDA)

if(CUDA_FOUND AND (NOT CUDA_VERSION VERSION_LESS 7.0))

    # GPU version

    file(GLOB SOURCE_GPU_FILES "*.cu")
    list(REMOVE_ITEM SOURCE_GPU_FILES "${CMAKE_SOURCE_DIR}/yolo_kernels.cu")

    set(CUDA_HOST_COMPILER "/usr/bin/gcc" CACHE STRING "host compiler" FORCE)

    include(FindCUDA)

    ## todo : check version
    # detect_cuDNN()

    message(STATUS "Cuda RT libs: ${CUDA_LIBRARIES}")
    message(STATUS "Cuda BLAS libs: ${CUDA_CUBLAS_LIBRARIES}")
    message(STATUS "Cuda Random lib: ${CUDA_curand_LIBRARY}")

    set(GPU_COMPILE_FLAGS "-DGPU=1")

    if(CUDNN_FOUND)
        include_directories(${CUDNN_INCLUDE})
        list(APPEND CUDA_NVCC_FLAGS -DCUDNN=1)
        list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARY})
        set(GPU_COMPILE_FLAGS "${GPU_COMPILE_FLAGS} -DCUDNN=1")
    endif()

    list(APPEND CUDA_NVCC_FLAGS -DGPU=1)

    #list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg -DGPU=1)

    set(CUDA_ARCH compute_30)

    if (DEFINED ENV{DPF_DARKNET_CUDA_ARCH})
        set(CUDA_ARCH $ENV{DPF_DARKNET_CUDA_ARCH})
    endif()

    message(STATUS "Darknet cuda arch: ${CUDA_ARCH}")

    list(APPEND CUDA_NVCC_FLAGS -gencode arch=${CUDA_ARCH},code=${CUDA_ARCH})

    cuda_add_library(${PROJECT_NAME}_gpu SHARED ${SOURCE_CPU_FILES} ${SOURCE_GPU_FILES})

    target_link_libraries(${PROJECT_NAME}_gpu pthread m ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${OpenCV_LIBS})

    set_target_properties(${PROJECT_NAME}_gpu PROPERTIES COMPILE_FLAGS ${GPU_COMPILE_FLAGS})

    install(TARGETS ${PROJECT_NAME}_gpu DESTINATION .)

endif()
