To run the benchmark do the following:

Build Embree:
  git clone https://github.com/embree/embree.git embree
  cd embree
  git checkout v2.9.0 (or any other release you want to benchmark)
  mkdir build
  cd build
  ccmake ..
  configure Embree to release mode, and use ICC, and enable AVX512KNL

Optix:
  - put optix into a folder called optix

Build and Run test:
  mkdir build
  ccmake ../code
  configure to release build, use ICC, and enable AVX512KNL
  - create a folder called build and build the code project in it
  - run ./run.sh

