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

/*! \addtogroup rivl_render_embree_ivl */
/*! @{ */

/*! Reflects a viewing vector V at a normal N. */
__device__ inline Sample3f reflect_(const float3 &V, const float3 &N) {
  float cosi = dot(V,N);
  return make_Sample3f(2.0f*cosi*N-V, 1.0f);
}

/*! Reflects a viewing vector V at a normal N. Cosine between V
 *  and N is given as input. */
__device__ inline Sample3f reflect_(const float3 &V, const float3 &N, const float cosi) {
  return make_Sample3f(2.0f*cosi*N-V, 1.0f);
}

// =======================================================
/*!Refracts a viewing vector V at a normal N using the relative
 *  refraction index eta. Eta is refraction index of outside medium
 *  (where N points into) divided by refraction index of the inside
 *  medium. The vectors V and N have to point towards the same side
 *  of the surface. The cosine between V and N is given as input and
 *  the cosine of -N and transmission ray is computed as output. */
__device__ inline Sample3f refract(const float3& V, const float3& N, const float eta, 
                        const float cosi, float &cost)
{
  const float k = 1.0f-eta*eta*(1.0f-cosi*cosi);
  if (k < 0.0f) { cost = 0.0f; return make_Sample3f(optix::make_float3(0.f),0.0f); }
  cost = sqrt(k);
  return make_Sample3f(eta*(cosi*N-V)-cost*N, sqr(eta));
}

/*! Computes fresnel coefficient for media interface with relative
 *  refraction index eta. Eta is the outside refraction index
 *  divided by the inside refraction index. Both cosines have to be
 *  positive. */
__device__ inline float fresnelDielectric(const float cosi, const float cost, const float eta)
{
  const float Rper = (eta*cosi -     cost) * rcp(eta*cosi +     cost);
  const float Rpar = (    cosi - eta*cost) * rcp(    cosi + eta*cost);
  return 0.5f*(Rpar*Rpar + Rper*Rper);
}

  /*! Computes fresnel coefficient for media interface with relative
   *  refraction index eta. Eta is the outside refraction index
   *  divided by the inside refraction index. The cosine has to be
   *  positive. */
__device__ inline float fresnelDielectric(const float cosi, const float eta)
{
  const float k = 1.0f-eta*eta*(1.0f-cosi*cosi);
  if (k < 0.0f) return 1.0f;
  const float cost = sqrt(k);
  return fresnelDielectric(cosi, cost, eta);
}

/*! Computes fresnel coefficient for conductor medium with complex
 *  refraction index (eta,k). The cosine has to be positive. */
__device__ inline float3 fresnelConductor(const float cosi, const float3& eta, const float3& k)
{
  const float3 tmp = eta*eta + k*k;
  const float3 Rpar = (tmp * (cosi*cosi) - 2.0f*eta*cosi + optix::make_float3(1.0f)) *
    rcp(tmp * (cosi*cosi) + 2.0f*eta*cosi + optix::make_float3(1.0f));
  const float3 Rper = (tmp - 2.0f*eta*cosi + optix::make_float3(cosi*cosi)) *
    rcp(tmp + 2.0f*eta*cosi + optix::make_float3(cosi*cosi));
  return 0.5f * (Rpar + Rper);
}

// =======================================================
struct FresnelConductor {
  float3 eta;  //!< Real part of refraction index
  float3 k;    //!< Imaginary part of refraction index
};

__device__ inline float3 eval(const FresnelConductor& This, const float cosTheta) {
  return fresnelConductor(cosTheta,This.eta,This.k);
}

__device__ inline FresnelConductor make_FresnelConductor(const float3& eta, const float3& k) {
  FresnelConductor m; m.eta = eta; m.k = k; return m;
}

#if defined(ISPC)
__device__ inline FresnelConductor make_FresnelConductor(const float3& eta, const float3& k) {
  FresnelConductor m; m.eta = eta; m.k = k; return m;
}
#endif

// =======================================================
struct FresnelDielectric 
{
  /*! refraction index of the medium the incident ray travels in */
  float etai;
  
  /*! refraction index of the medium the outgoing transmission rays
   *  travels in */
  float etat;
};

__device__ inline float3 eval(const FresnelDielectric& This, const float cosTheta) {
  return optix::make_float3(fresnelDielectric(cosTheta,This.etai/This.etat));
}

__device__ inline FresnelDielectric make_FresnelDielectric(const float etai, const float etat) {
  FresnelDielectric m; m.etai = etai; m.etat = etat; return m;
}

#if defined(ISPC)
__device__ inline FresnelDielectric make_FresnelDielectric(const float etai, const float etat) {
  FresnelDielectric m; m.etai = etai; m.etat = etat; return m;
}
#endif

// =======================================================
struct PowerCosineDistribution {
  float exp;
};

__device__ inline float eval(const PowerCosineDistribution &This, const float cosThetaH) {
  return (This.exp+2) * (1.0f/(2.0f*(pi))) * pow(abs(cosThetaH), This.exp);
}

#if defined(ISPC)

__device__ inline float eval(const PowerCosineDistribution &This, const float cosThetaH) {
  return (This.exp+2) * (1.0f/(2.0f*(pi))) * pow(abs(cosThetaH), This.exp);
}
#endif

/*! Samples the power cosine distribution. */
__device__ inline void sample(const PowerCosineDistribution& This, const float3& wo, const float3& N, Sample3f &wi, const float2 s)  
{
  Sample3f wh = powerCosineSampleHemisphere(s.x,s.y,N,This.exp);
  Sample3f r = reflect_(wo,wh.v);
  wi = make_Sample3f(r.v,wh.pdf/(4.0f*abs(dot(wo,wh.v))));
}

/*! Samples the power cosine distribution. */
#if defined(ISPC)
__device__ inline void sample(const PowerCosineDistribution& This, const float3& wo, const float3& N, Sample3f &wi, const float2 s)  
{
  Sample3f wh = powerCosineSampleHemisphere(s.x,s.y,N,This.exp);
  Sample3f r = reflect_(wo,wh.v);
  wi = make_Sample3f(r.v,wh.pdf/(4.0f*abs(dot(wo,wh.v))));
}
#endif

__device__ inline PowerCosineDistribution make_PowerCosineDistribution(const float _exp) { 
  PowerCosineDistribution m; m.exp = _exp; return m;
}

#if defined(ISPC)
__device__ inline PowerCosineDistribution make_PowerCosineDistribution(const float _exp) { 
  PowerCosineDistribution m; m.exp = _exp; return m;
}
#endif

/*! @} */
