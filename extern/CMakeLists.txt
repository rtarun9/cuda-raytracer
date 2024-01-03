add_library(extern STATIC stb/stb_image_write.h stb/stb_image_write.cpp)

target_include_directories(extern PUBLIC ".")