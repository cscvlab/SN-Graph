CMAKE_MINIMUM_REQUIRED_VERSION(VERSION 3.10)

PROJECT(SNG)

aux_source_directory()(./ DIR_SRCS)

add_executable(sng.exe ${DIR_SRCS})

target_link_libraries(demo3 edt)