// ======================================================================== //
// Copyright 2009-2015 Intel Corporation                                    //
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

#pragma once

/*! \file shapesampler.isph Implements sampling functions for different
 *  geometric shapes. */

//__device__ inline float cos2sin(const float f) { return sqrt(max(0.f,1.f-f*f)); }
//__device__ inline float sin2cos(const float f) { return sqrt(max(0.f,1.f-f*f)); }

/*! Cosine weighted hemisphere sampling. Up direction is the z direction. */
__device__ inline Sample3f cosineSampleHemisphere(const float u, const float v) {
  const float phi = 2.0f * (pi) * u;
  const float cosTheta = sqrt(v);
  const float sinTheta = sqrt(1.0f - v);
  return make_Sample3f(optix::make_float3(cos(phi) * sinTheta, 
                                  sin(phi) * sinTheta, 
                                  cosTheta), 
                       cosTheta*(1.f/(pi)));
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument. */
__device__ inline Sample3f cosineSampleHemisphere(const float  u, const float  v, const float3& N) 
{
  Sample3f s = cosineSampleHemisphere(u,v);
  return make_Sample3f(frame(N)*s.v,s.pdf);
}

  /*! Samples hemisphere with power cosine distribution. Up direction
   *  is the z direction. */
__device__ inline Sample3f powerCosineSampleHemisphere(const float u, const float v, const float _exp) 
{
  const float phi = 2.0f * (pi) * u;
  const float cosTheta = pow(v,1.0f/(_exp+1.0f));
  const float sinTheta = cos2sin(cosTheta);
  return make_Sample3f(optix::make_float3(cos(phi) * sinTheta, 
				   sin(phi) * sinTheta, 
				   cosTheta), 
                       (_exp+1.0f)*pow(cosTheta,_exp)*0.5f/(pi));
}

/*! Computes the probability density for the power cosine sampling of the hemisphere. */
__device__ inline float powerCosineSampleHemispherePDF(const float3& s, const float _exp) {
  if (s.z < 0.f) return 0.f;
  return (_exp+1.0f)*pow(s.z,_exp)*0.5f/pi;
}

/*! Samples hemisphere with power cosine distribution. Up direction
 *  is provided as argument. */
__device__ inline Sample3f powerCosineSampleHemisphere(const float u, const float v, const float3& N, const float _exp) {
  Sample3f s = powerCosineSampleHemisphere(u,v,_exp);
  return make_Sample3f(frame(N)*s.v,s.pdf);
}

////////////////////////////////////////////////////////////////////////////////
/// Sampling of Spherical Cone
////////////////////////////////////////////////////////////////////////////////


/*! Uniform sampling of spherical cone. Cone direction is the z
 *  direction. */
__device__ inline Sample3f UniformSampleCone(const float u, const float v, const float angle) {
  const float phi = (float)(2.0f * pi) * u;
  const float cosTheta = 1.0f - v*(1.0f - cos(angle));
  const float sinTheta = cos2sin(cosTheta);
  return make_Sample3f(optix::make_float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta), 1.0f/((float)(4.0f*pi)*sqr(sin(0.5f*angle))));
}

/*! Computes the probability density of spherical cone sampling. */
__device__ inline float UniformSampleConePDF(const float3 &s, const float angle) {
  return select(s.z < cos(angle), 0.0f, 1.0f/((float)(4.0f*pi)*sqr(sin(0.5f*angle))));
}

/*! Uniform sampling of spherical cone. Cone direction is provided as argument. */
__device__ inline Sample3f UniformSampleCone(const float u, const float v, const float angle, const float3& N) {
  Sample3f s = UniformSampleCone(u,v,angle);
  return make_Sample3f(frame(N)*s.v,s.pdf);
}

/*! Computes the probability density of spherical cone sampling. */
__device__ inline float UniformSampleConePDF(const float3 &s, const float angle, const float3 &N) {
  // return make_select(dot(s,N) < cos(angle), 0.0f, 1.0f/((float)(4.0f*pi)*sqr(sin(0.5f*angle))));
  if (dot(s,N) < cos(angle))
    return 0.f;
  else
    return 1.0f/((float)(4.0f*pi)*sqr(sin(0.5f*angle)));
}

////////////////////////////////////////////////////////////////////////////////
/// Sampling of Triangle
////////////////////////////////////////////////////////////////////////////////

/*! Uniform sampling of triangle. */
__device__ inline float3 UniformSampleTriangle(const float u, const float v, const float3& A, const float3& B, const float3& C) {
  const float su = sqrt(u);
  return C + (1.0f-su)*(A-C) + (v*su)*(B-C);
}

////////////////////////////////////////////////////////////////////////////////
/// Sampling of Disk
////////////////////////////////////////////////////////////////////////////////

/*! Uniform sampling of disk. */
__device__ inline float2 UniformSampleDisk(const float2 &sample, const float radius) 
{
  const float r = sqrt(sample.x);
  const float theta = (2.f*pi) * sample.y;
  return make_float2(radius*r*cos(theta), radius*r*sin(theta));
}
