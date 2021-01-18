cmake_minimum_required(VERSION 3.8)
project(RegistrationTools)

#enable c++11
set(CMAKE_CXX_STANDARD 14)

include_directories(src/)

#fine CUDA:
enable_language(CUDA)
find_package(CUDA)

#Setting yaml-cpp:
if(WIN32)
set(yaml-cpp_DIR "C:/yaml-cpp/install/share/cmake/yaml-cpp")
else()
# set(yaml-cpp_DIR "/home/wenhai/packages/yaml-cpp/build")
endif()
find_package(yaml-cpp REQUIRED)
if(WIN32)
# add_definitions(-DYAML_CPP_DLL)
include_directories(C:/yaml-cpp/install/include)
# include_directories(RegistrationTools C:/Users/wenhai/Documents/libs/yaml-cpp-master/build/Debug)
endif()

#OpenIgtLink:
set(OpenIGTLink_DIR C:/OpenIGTLink/build)
find_package(OpenIGTLink REQUIRED)
include_directories(RegistrationTools ${OpenIGTLink_INCLUDE_DIRS})

include_directories(RegistrationTools ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${YAML_CPP_INCLUDE_DIR})
include_directories(RegistrationTools include)

#ITK: 
set(ITK_DIR C:/ITK/build)
find_package(ITK COMPONENTS REQUIRED)
include_directories(${ITK_INCLUDE_DIRS})

#nlopt: 
set(nlopt_DIR C:/nlopt/build)
find_package(nlopt REQUIRED)
include_directories(RegistrationTools ${NLOPT_INCLUDE_DIRS})

add_executable(
    CMS_RegistrationTools

    src/mainRegistrationTool/main.cu
    src/mainRegistrationTool/md5.cpp
    src/cu_LC2/cu_lc2_3d.cu
    src/cu_LC2/cu_lc2_2d.cu
    src/Interpolator/interpolator.cu
    src/Interpolator/spatialMappers.cu
    src/Interpolator/volumeImporter.cpp
    src/Optimizers/optimizers.cu
    src/Utilities/my_cuda_helper.cu
)

target_link_libraries(
    CMS_RegistrationTools 
    C:/yaml-cpp/install/lib/yaml-cpp.lib
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_nppist_LIBRARY}
    ${NLOPT_LIBRARIES}
    ${OpenIGTLink_LIBRARIES}
    ${ITK_LIBRARIES}
)
