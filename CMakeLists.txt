cmake_minimum_required(VERSION 3.10)
project(gpupreprocessor)

include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)
#@findhao: for debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Default build type: RelWithDebInfo" FORCE)
endif()


if(${CMAKE_VERSION} VERSION_LESS "3.17.0") 
    find_package(CUDA REQUIRED)
    set(CUDAToolkit_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    set(CUDAToolkit_TARGET_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    if( ${CUDA_VERSION} LESS 11.2 )
        message(FATAL_ERROR "CUDAToolkit mush be newer or equal to 11.2. Your version is " ${CUDA_VERSION} )
    endif()

else()
    find_package(CUDAToolkit REQUIRED)
    if (NOT CMAKE_CUDA_COMPILER)
        if (NOT ${CUDAToolkit_NVCC_EXECUTABLE} AND UNIX AND EXISTS "/usr/local/cuda/bin/nvcc")
            set(CUDAToolkit_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
            message(WARNING "CMAKE_CUDA_COMPILER guessed: " ${CUDAToolkit_NVCC_EXECUTABLE} "\n"
                    "Please fix your cuda installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-post")
        endif ()
    endif ()
    if( ${CUDAToolkit_VERSION} LESS 11.2 )
        message(FATAL_ERROR "CUDAToolkit mush be newer or equal to 11.2. Your version is " ${CUDAToolkit_VERSION} )
    endif()
endif()


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS " -Wall -Wextra")


include_directories(${CUDAToolkit_TARGET_DIR}/include/ 
                ${CMAKE_CURRENT_LIST_DIR}/include/
                ${CMAKE_CURRENT_LIST_DIR}/src/
                /home/yhao24/NVIDIA_CUDA-11.4_Samples/common/inc/
                )
message(INFO  ${CUDAToolkit_NVCC_EXECUTABLE})


# add_custom_target(gputrigger_patch.fatbin ALL ${CUDAToolkit_NVCC_EXECUTABLE} -o gputrigger_patch.fatbin ${CMAKE_CUDA_FLAGS} --fatbin --keep-device-functions -Xptxas --compile-as-tools-patch -g  -I${SANITIZER_PATH}/include/ -I${CUDAToolkit_TARGET_DIR}/include/ -c ${CMAKE_CURRENT_LIST_DIR}/gpu-patch.cu )
# set_property(TARGET gputrigger_patch.fatbin PROPERTY CUDA_ARCHITECTURES "70;75;80;86")

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/ gputrigger_SRC)
file(GLOB gputrigger_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/*.h" )
add_executable(gpupreprocessor ${gputrigger_SRC} ${gputrigger_headers} )

target_link_libraries(gpupreprocessor PUBLIC
        -lcuda
        -lcudart
        -L${CUDAToolkit_TARGET_DIR}/lib64
        )

# install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gputrigger_patch.fatbin DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS gpupreprocessor DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES  ${gputrigger_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
