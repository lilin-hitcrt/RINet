set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
if (NOT PCL_FOUND)
    find_package(PCL REQUIRED)
endif ()
if (NOT OPENCV_FOUND)
    find_package(OpenCV  REQUIRED)
endif ()
find_package (yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
file(GLOB SRC_LIST *.cpp)
add_library(gencloud SHARED ${SRC_LIST})
target_include_directories(gencloud PUBLIC 
${PCL_INCLUDE_DIRS} 
${EIGEN3_INCLUDE_DIRS}
./
)
if (NOT YAML_CPP_LIBRARIES)
    set(YAML_CPP_LIBRARIES yaml-cpp)
endif ()
target_link_libraries(gencloud
        ${PCL_LIBRARIES}
${YAML_CPP_LIBRARIES}
${OpenCV_LIBS}
)




