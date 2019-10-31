To run the benchmark do the following:

Build Embree:
  git clone https://github.com/embree/embree.git embree
  cd embree
  git checkout v3.5.2 (or any other release you want to benchmark)
  mkdir build
  cd build
  ccmake ..
  configure Embree to release mode, and use ICC
  make

Optix:
  - put OptiX-SDK-6.0.0 into a folder called optix 

Build and Run test:
  mkdir build
  ccmake ../code
  configure to release build, use ICC
  make
  cd ..
  run ./run.sh

