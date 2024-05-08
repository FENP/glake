set(SRC CUDACachingAllocator.cpp)

add_library(${LIB_NAME} SHARED ${SRC})

target_include_directories(${LIB_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${PYTORCH_DIR}/include
)
target_link_directories(${LIB_NAME} PRIVATE ${PYTORCH_DIR}/lib)

target_link_libraries(${LIB_NAME} PRIVATE cuda c10 c10_cuda)

set_target_properties(${LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)