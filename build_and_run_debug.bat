cmake -S . -B build -G "Ninja" 
cmake --build build --config Debug

build\src\cuda-raytracer.exe
