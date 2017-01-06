// ======================================================================== //
// Copyright 2009-2013 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define COUNT_RAYS(x)
//#define COUNT_RAYS(x) x

#include <optix.h>
#include <optix_math.h>
#include <optixu/optixu_aabb.h>

/* some functions from math.h */
#define CUDA_INF __int_as_float(0x7f800000)
#define pi 3.14159265359f
#define one_over_pi (1.0f/pi)
#define one_over_two_pi (1.0f/(2.0f*pi))
__device__ inline float3 neg(float3 a) { return -a; }
__device__ inline float4 neg(float4 a) { return -a; }
__device__ inline float sin2cos ( const float x )  { return sqrt(max(0.0f,1.0f-x*x)); }
__device__ inline float cos2sin ( const float x )  { return sin2cos(x); }
__device__ inline float sqr ( const float x )  { return x*x; }
__device__ inline float rcp ( const float x )  { return 1.0f/x; }
__device__ inline float3 rcp ( const float3 v )  { return 1.0f/v; }
__device__ inline float clamp ( const float x )  { return optix::clamp(x,0.0f,1.0f); }
__device__ inline float select (const bool b, const float t, const float f) { return b ? t : f; }
__device__ inline bool eq ( const float3 a, const float3 b )  { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
__device__ inline bool ne ( const float3 a, const float3 b )  { return (a.x != b.x) || (a.y != b.y) || (a.z == b.z); }
__device__ inline float3 pow ( const float3 a, const float b )  { return optix::make_float3(pow(a.x,b),pow(a.y,b),pow(a.z,b)); }
__device__ inline float3 exp ( const float3 a )  { return optix::make_float3(exp(a.x),exp(a.y),exp(a.z)); }

/* LinearSpace3f implementation for OptiX */

struct LinearSpace3f 
{
  float3 vx;
  float3 vy;
  float3 vz;
};

__device__ inline LinearSpace3f make_LinearSpace3f(const float3 x, const float3 y, const float3 z) { 
  LinearSpace3f l; l.vx = x; l.vy = y; l.vz = z; return l; 
}

__device__ inline LinearSpace3f frame(const float3 N) 
{
  const float3 dx0 = cross(make_float3(1.0f,0.0f,0.0f),N);
  const float3 dx1 = cross(make_float3(0.0f,1.0f,0.0f),N);
  const float3 dx = normalize(dot(dx0,dx0) > dot(dx1,dx1) ? dx0 : dx1);
  const float3 dy = normalize(cross(N,dx));
  return make_LinearSpace3f(dx,dy,N);
}

__device__ inline float3 operator*(const LinearSpace3f l, const float3 v) { 
  return v.x*l.vx + v.y*l.vy + v.z*l.vz; 
}

/* Sample3f implemantation for OptiX */
struct Sample3f
{
  float3 v;
  float pdf;
};

__device__ inline Sample3f make_Sample3f(const float3 v, const float pdf) {
  Sample3f s; s.v = v; s.pdf = pdf; return s;
}

/* some helper functions for proper ray creation */
__device__ inline optix::Ray make_Ray(float3 org,float3 dir, float tnear, float tfar) {
  return optix::make_Ray(org,dir,0,tnear,tfar);
}

__device__ inline optix::Ray make_ShadowRay(float3 org,float3 dir, float tnear, float tfar) {
  return optix::make_Ray(org,dir,1,tnear,tfar);
}

/* scene geometry */
rtBuffer<int3> triangles;
rtBuffer<int> trimaterialID;
rtBuffer<float3> positions;  

/* framebuffer */
rtBuffer<uchar4, 2> frameBuffer;
rtBuffer<int, 1> raycounter;

/* pixel of current invokation */
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

/* camera parameters */
rtDeclareVariable(float3, P, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

/* scene */
rtDeclareVariable(rtObject, g_scene, , );

struct Hit
{
  float u;
  float v;
  float t;
  float3 Ng;
  int primID;
};
rtDeclareVariable(Hit, payload,  rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void triangle_intersect(int id)
{
  int3 tri = triangles[id];
  float3 p0 = positions[tri.x];
  float3 p1 = positions[tri.y];
  float3 p2 = positions[tri.z];

#if 1
  float3 Ng; 
  float t,u,v;
  if (intersect_triangle_branchless(ray,p0,p1,p2,Ng,t,u,v))
  //if (intersect_triangle_earlyexit(ray,p0,p1,p2,Ng,t,u,v))
  {
    if (rtPotentialIntersection(t)) {
      payload.u = u;
      payload.v = v;
      payload.t = t;
      payload.Ng = Ng;
      payload.primID = id;
      rtReportIntersection(0);
    }
  }
#else
  float3 E1 = p1 - p0;
  float3 E2 = p2 - p0;
  float3 T = ray.origin - p0;
  float3 P = cross(ray.direction,E2);
  float3 Q = cross(T,E1);
  float det = dot(P,E1);
  if (det == 0.0f) return;
  float rcp_det = 1.0f/det;
  float t = dot(Q,E2) * rcp_det;
  float u = dot(P,T) * rcp_det;
  float v = dot(Q,ray.direction) * rcp_det;
  if (min(min(u,v),1-u-v) < 0.0f) return;
  if (rtPotentialIntersection(t)) {
    payload.u = u;
    payload.v = v;
    payload.t = t;
    payload.Ng = cross(E2,E1);
    payload.primID = id;
    rtReportIntersection(0);
  }
#endif
}

/* bounding box program for building acceleration structure */
RT_PROGRAM void triangle_bounds (int id, float result[6])
{
  int3 tri = triangles[id];
  float3 p0 = positions[tri.x];
  float3 p1 = positions[tri.y];
  float3 p2 = positions[tri.z];
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = fminf( fminf( p0, p1), p2 );
  aabb->m_max = fmaxf( fmaxf( p0, p1), p2 );
}

typedef int2 Vec2i;
#define g_accu_count 0

#include "pathtracer_device.cu"

/* render frame by shooting primary rays */
RT_PROGRAM void render_primary()
{
  int x = launch_index.x;
  int y = launch_index.y;
  float3 L = renderPixelStandard(x,y,U,V,W,P);
  frameBuffer[launch_index] = make_uchar4(__saturatef(L.x)*255.99f,__saturatef(L.y)*255.99f,__saturatef(L.z)*255.99f,255u);
}

RT_PROGRAM void miss_program()
{
}

RT_PROGRAM void closest_hit_program()
{
}

RT_PROGRAM void any_hit_program()
{
  rtTerminateRay();
}
