#!/bin/bash

mkdir -p stat
#./benchmark.py run single ./build/pathtracer
#./benchmark.py run ispc ./build/pathtracer_ispc
#./benchmark.py print single ispc
#./benchmark.py run optix ./build/pathtracer_optix
./benchmark.py print single ispc optix
