
all:
	@./ispc2optix.sh pathtracer/pathtracer_device.ispc pathtracer/pathtracer_device.cu
	@./ispc2optix.sh pathtracer/shapesampler.isph pathtracer/shapesampler.cu.h
	@./ispc2optix.sh pathtracer/optics.isph pathtracer/optics.cu.h

	@./ispc2cpp.sh pathtracer/pathtracer_device.ispc pathtracer/pathtracer_device.cpp
	@./ispc2cpp.sh pathtracer/shapesampler.isph pathtracer/shapesampler.h
	@./ispc2cpp.sh pathtracer/optics.isph pathtracer/optics.h
