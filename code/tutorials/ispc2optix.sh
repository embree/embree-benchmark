#!/bin/bash

echo Converting ISPC tutorial $1 to Optix tutorial $2
cp $1 $2

sed -i.backup  's/unmasked //g' $2
sed -i.backup  's/uniform //g' $2
sed -i.backup  's/ uniform//g' $2
sed -i.backup  's/varying //g' $2
sed -i.backup  's/\#include \"\.\.\/common\/tutorial\/tutorial_device.isph\"//g' $2
sed -i.backup  's/\#include \"\.\.\/common\/tutorial\/scene_device.isph\"/\#include \"\.\.\/common\/tutorial\/scene_device.cu.h\"/g' $2
sed -i.backup  's/\#include \"shapesampler.isph\"/\#include \"shapesampler.cu.h\"/g' $2
sed -i.backup  's/\#include \"optics.isph\"/\#include \"optics.cu.h\"/g' $2
sed -i.backup  's/RTCRay/optix::Ray/g' $2

sed -i.backup  's/inline/__device__ inline/g' $2
sed -i.backup  's/Vec3fa/float4/g' $2
sed -i.backup  's/Vec3f/float3/g' $2
sed -i.backup  's/Vec2f/float2/g' $2
sed -i.backup  's/inf/CUDA_INF/g' $2
sed -i.backup  's/make_float3/optix::make_float3/g' $2

sed -i.backup  's/renderPixelFunc renderPixel/\/\/renderPixelFunc renderPixel/g' $2

sed -i.backup  's/#if !defined(CODE_DISABLED)/#if 0/g' $2

sed -i.backup  's/getMaterialID(ray,dg)/materialID = trimaterialID[hit.primID]/g' $2

sed -i.backup  's/rtcIntersect(g_scene,ray);/hit.u = hit.v = hit.t = 0.0f; hit.Ng = optix::make_float3(0,0,0); hit.primID = -1; rtTrace(g_scene, ray, hit);/g' $2
sed -i.backup  's/rtcOccluded(g_scene,shadow);/Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);/g' $2

sed -i.backup  's/optix::Ray ray/Hit hit; optix::Ray ray/g' $2

sed -i.backup  's/ray.dir/ray.direction/g' $2
sed -i.backup  's/ray.org/ray.origin/g' $2
sed -i.backup  's/ray.tfar/hit.t/g' $2
sed -i.backup  's/ray.u/hit.u/g' $2
sed -i.backup  's/ray.v/hit.v/g' $2
sed -i.backup  's/ray.Ng/hit.Ng/g' $2
sed -i.backup  's/ray.geomID/hit.primID/g' $2
sed -i.backup  's/ray.primID/hit.primID/g' $2
sed -i.backup  's/shadow.geomID/shit.primID/g' $2
sed -i.backup  's/shadow.primID/shit.primID/g' $2
sed -i.backup  's/RTC_INVALID_GEOMETRY_ID/-1/g' $2

sed -i.backup  's/g_ispc_scene->meshes\[ray.geomID\]->triangles\[ray.primID\].materialID/trimaterialID\[hit.primID\]/g' $2
sed -i.backup  's/g_ispc_scene->materials/materials/g' $2

sed -i.backup  's/float3 renderPixelFunction/__device__ float3 renderPixelFunction/g' $2
sed -i.backup  's/float3 renderPixelStandard/__device__ float3 renderPixelStandard/g' $2

sed -i.backup  's/g_ispc_scene->numAmbientLights/numAmbientLights/g' $2
sed -i.backup  's/g_ispc_scene->numPointLights/numPointLights/g' $2
sed -i.backup  's/g_ispc_scene->numDirectionalLights/numDirectionalLights/g' $2
sed -i.backup  's/g_ispc_scene->numDistantLights/numDistantLights/g' $2

sed -i.backup  's/unsigned int g_accu_count/__device__ unsigned int g_accu_count/g' $2

# rewrite scene
sed -i.backup  's/struct ISPCScene {//g' $2
sed -i.backup  's/ISPCMesh\*\* meshes;/\/\/ISPCMesh\*\* meshes;/g' $2
sed -i.backup  's/ISPCMaterial\* materials;/rtBuffer<ISPCMaterial> materials; \/\//g' $2

sed -i.backup  's/ISPCAmbientLight\* ambientLights;/rtBuffer<ISPCAmbientLight> ambientLights;/g' $2
sed -i.backup  's/int numAmbientLights;/rtDeclareVariable(int, numAmbientLights, , );/g' $2
sed -i.backup  's/g_ispc_scene->ambientLights\[i\]/ambientLights\[i\]/g' $2

sed -i.backup  's/ISPCPointLight\* pointLights;/rtBuffer<ISPCPointLight> pointLights;/g' $2
sed -i.backup  's/int numPointLights;/rtDeclareVariable(int, numPointLights, , );/g' $2
sed -i.backup  's/g_ispc_scene->pointLights\[i\]/\pointLights\[i\]/g' $2

sed -i.backup  's/ISPCDirectionalLight\* dirLights;/rtBuffer<ISPCDirectionalLight> dirLights;/g' $2
sed -i.backup  's/int numDirectionalLights;/rtDeclareVariable(int, numDirectionalLights, , );/g' $2
sed -i.backup  's/g_ispc_scene->dirLights\[i\]/\*(ISPCDirectionalLight\*)\&dirLights\[i\]/g' $2

sed -i.backup  's/ISPCDistantLight\* distantLights;/rtBuffer<ISPCDistantLight> distantLights;/g' $2
sed -i.backup  's/int numDistantLights;/rtDeclareVariable(int, numDistantLights, , );/g' $2
sed -i.backup  's/g_ispc_scene->distantLights\[i\]/\*(ISPCDistantLight\*)\&distantLights\[i\]/g' $2

sed -i.backup  's/}; \/\/ ISPCScene//g' $2

sed -i.backup  's/RTCScene g_scene/\/\/RTCScene g_scene/g' $2
sed -i.backup  's/extern ISPCScene\* g_ispc_scene;/\/\/extern ISPCScene\* g_ispc_scene;/g' $2

sed -i.backup  's/if (id < 0 || id >= numMaterials) continue;//g' $2
sed -i.backup  's/foreach_unique (id in materialID)//g' $2
sed -i.backup  's/ISPCMaterial\* material = \&materials\[id\];/ISPCMaterial\* material = \&materials\[materialID\];/g' $2
sed -i.backup  's/\#define __device__//g' $2
sed -i.backup  's/g_ispc_scene->numMaterials/0/g' $2

sed -i.backup  's/M_PI/pi/g' $2
