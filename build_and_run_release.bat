cmake -S . -B build -G "Ninja" 
cmake --build build --config Release 

build\src\cuda-raytracer.exe
