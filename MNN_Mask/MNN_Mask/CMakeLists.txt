cmake_minimum_required(VERSION 2.8)
project(FaceMask)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  

add_definitions(-Wno-unused-command-line-argument)
add_definitions(-Wall)
add_definitions(-fPIC)
add_definitions(-g)
#add_definitions(-O3)
add_definitions(-funroll-loops)
add_definitions(-Wno-overloaded-virtual)
add_definitions(-Wno-deprecated-register)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)
set( MNN_DIR /usr/local/)
set( MNN_LIBS MNN MNN_Express)
set( CODE_SRCS FaceMask_MNN.cpp )
set( BIN_EXE FaceMask_MNN )
set( ThirdParty /home/hyliu/MNN/3rd_party/imageHelper)
#set( ROCKCHIP_LIB /usr/lib64/librockchip_vendor.a )

find_package(OpenCV REQUIRED)

#include
include_directories(${MNN_DIR}/include/MNN
                    ${ThirdParty})
#lib
link_directories(${MNN_DIR}/lib
                )

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
#target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${ROCKCHIP_LIB} ${OpenCV_LIBS} pthread dl)
target_link_libraries(${BIN_EXE} ${MNN_LIBS} ${OpenCV_LIBS})
