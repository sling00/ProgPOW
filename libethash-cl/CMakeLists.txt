# A custom command and target to turn the OpenCL kernel into a byte array header
# The normal build depends on it properly and if the kernel file is changed, then
# a rebuild of libethash-cl should be triggered

#TODO: clean up the copy&pasting here
add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ethash.h
	COMMAND ${CMAKE_COMMAND} ARGS
	-DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/kernels/cl/ethash.cl"
	-DBIN2H_VARIABLE_NAME=ethash_cl
	-DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/ethash.h"
	-P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/bin2h.cmake"
	COMMENT "Generating OpenCL Ethash Kernel Byte Array"
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cl/ethash.cl
)
add_custom_target(clbin2h_ethash DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ethash.h ${CMAKE_CURRENT_SOURCE_DIR}/cl/ethash.cl)

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/progpow.h
	COMMAND ${CMAKE_COMMAND} ARGS
	-DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/kernels/cl/progpow.cl"
	-DBIN2H_VARIABLE_NAME=progpow_cl
	-DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/progpow.h"
	-P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/bin2h.cmake"
	COMMENT "Generating OpenCL ProgPoW Kernel Byte Array"
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cl/progpow.cl
)
add_custom_target(clbin2h_progpow DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/progpow.h ${CMAKE_CURRENT_SOURCE_DIR}/cl/progpow.cl)


set(SOURCES
	CLMiner.h CLMiner.cpp
	${CMAKE_CURRENT_BINARY_DIR}/ethash.h
	${CMAKE_CURRENT_BINARY_DIR}/progpow.h
)

if(APPLE)
	# On macOS use system OpenCL library.
	find_package(OpenCL REQUIRED)
else()
	hunter_add_package(OpenCL)
	find_package(OpenCL CONFIG REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(..)

add_library(ethash-cl ${SOURCES})
target_link_libraries(ethash-cl PUBLIC ethcore ethash::ethash)
target_link_libraries(ethash-cl PUBLIC ethcore progpow)
target_link_libraries(ethash-cl PRIVATE OpenCL::OpenCL)
target_link_libraries(ethash-cl PRIVATE Boost::filesystem)
