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


#include "../common/tutorial/scene_device.cu.h"
#include "shapesampler.cu.h"
#include "optics.cu.h"

#define COUNT_RAYS(x)
COUNT_RAYS(size_t numRays = 0);



#undef TILE_SIZE_X
#undef TILE_SIZE_Y

#define TILE_SIZE_X 4
#define TILE_SIZE_Y 4

#define FIX_SAMPLING 1
#define SAMPLES_PER_PIXEL 1

#define ENABLE_TEXTURING 0
#define ENABLE_TEXTURE_COORDINATES 0
#define ENABLE_OCCLUSION_FILTER 0
#define ENABLE_DISPLACEMENTS 0

//#define FORCE_FIXED_EDGE_TESSELLATION
#define FIXED_EDGE_TESSELLATION_VALUE 4

#define MAX_EDGE_LEVEL 128.0f
#define MIN_EDGE_LEVEL   4.0f
#define LEVEL_FACTOR    64.0f
#define MAX_PATH_LENGTH  8

bool g_subdiv_mode = false;
unsigned int keyframeID = 0;

struct DifferentialGeometry
{
  int geomID;
  int primID;
  float u,v;
  float3 P;
  float3 Ng;
  float3 Ns;
};

struct BRDF
{
  float Ns;               /*< specular exponent */
  float Ni;               /*< optical density for the surface (index of refraction) */
  float3 Ka;              /*< ambient reflectivity */
  float3 Kd;              /*< diffuse reflectivity */
  float3 Ks;              /*< specular reflectivity */
  float3 Kt;              /*< transmission filter */
};

struct Medium
{
  float3 transmission; //!< Transmissivity of medium.
  float eta;             //!< Refraction index of medium.
};

__device__ inline Medium make_Medium(const float3& transmission, const float eta)
{
  Medium m;
  m.transmission = transmission;
  m.eta = eta;
  return m;
}

__device__ inline Medium make_Medium_Vacuum() { 
  return make_Medium(optix::make_float3((float)1.0f),1.0f); 
}

__device__ inline bool eq(const Medium& a, const Medium& b) {
  return (a.eta == b.eta) && eq(a.transmission, b.transmission);
}

__device__ inline float3 sample_component2(const float3& c0, const Sample3f& wi0, const Medium& medium0,
                               const float3& c1, const Sample3f& wi1, const Medium& medium1,
                               const float3& Lw, Sample3f& wi_o, Medium& medium_o, const float s)
{
  const float3 m0 = Lw*c0/wi0.pdf;
  const float3 m1 = Lw*c1/wi1.pdf;

  const float C0 = wi0.pdf == 0.0f ? 0.0f : max(max(m0.x,m0.y),m0.z);
  const float C1 = wi1.pdf == 0.0f ? 0.0f : max(max(m1.x,m1.y),m1.z);
  const float C  = C0 + C1;

  if (C == 0.0f) {
    wi_o = make_Sample3f(optix::make_float3(0,0,0),0);
    return optix::make_float3(0,0,0);
  }

  const float CP0 = C0/C;
  const float CP1 = C1/C;
  if (s < CP0) {
    wi_o = make_Sample3f(wi0.v,wi0.pdf*CP0); 
    medium_o = medium0; return c0;
  } 
  else {
    wi_o = make_Sample3f(wi1.v,wi1.pdf*CP1); 
    medium_o = medium1; return c1;
  }
}

////////////////////////////////////////////////////////////////////////////////
//                             Ambient Light                                  //
////////////////////////////////////////////////////////////////////////////////

__device__ inline float3 AmbientLight__eval(const ISPCAmbientLight& light, const float3& wo) {
  return optix::make_float3(light.L);
}

__device__ inline float3 AmbientLight__sample(const ISPCAmbientLight& light, const DifferentialGeometry& dg, Sample3f& wi, float& tMax, const float2& s) 
{
  wi = cosineSampleHemisphere(s.x,s.y,dg.Ns);
  tMax = 1e20f;
  return optix::make_float3(light.L);
}

////////////////////////////////////////////////////////////////////////////////
//                             Point Light                                    //
////////////////////////////////////////////////////////////////////////////////

__device__ inline float3 PointLight__sample(const ISPCPointLight& light, 
					const DifferentialGeometry& dg, 
					Sample3f& wi,
					float& tMax,
					const float2& s) 
{
  float3 d = optix::make_float3(light.P) - dg.P;
  float distance = length(d);
  wi = make_Sample3f(d*rcp(distance), distance*distance);
  tMax = distance;
  return optix::make_float3(light.I);
}

////////////////////////////////////////////////////////////////////////////////
//                        Directional Light                                   //
////////////////////////////////////////////////////////////////////////////////

__device__ inline float3 DirectionalLight__sample(const ISPCDirectionalLight& light, 
					      const DifferentialGeometry& dg, 
					      Sample3f& wi,
					      float& tMax,
					      const float2& s) 
{
  wi = make_Sample3f(neg(normalize(optix::make_float3(light.D))),1.0f); 
  tMax = CUDA_INF; 
  return optix::make_float3(light.E);
}

////////////////////////////////////////////////////////////////////////////////
//                          Distant Light                                     //
////////////////////////////////////////////////////////////////////////////////

__device__ inline float3 DistantLight__eval(const ISPCDistantLight& light, const float3& wo) 
{
  if (-dot(wo,optix::make_float3(light.D)) >= light.cosHalfAngle) return optix::make_float3(light.L);
  return optix::make_float3(0.0f);
}

__device__ inline float3 DistantLight__sample(const ISPCDistantLight& light,
                                   const DifferentialGeometry& dg, 
                                   Sample3f& wi,
                                   float& tMax,
                                   const float2& s) 
{
  wi = UniformSampleCone(s.x,s.y,light.radHalfAngle,optix::make_float3((float4)neg(light.D)));
  tMax = 1e20f;

  return optix::make_float3(light.L);
}

////////////////////////////////////////////////////////////////////////////////
//                          Minneart BRDF                                     //
////////////////////////////////////////////////////////////////////////////////

struct Minneart
{
  /*! The reflectance parameter. The vale 0 means no reflection,
   *  and 1 means full reflection. */
  float3 R;
  
  /*! The amount of backscattering. A value of 0 means lambertian
   *  diffuse, and CUDA_INF means maximum backscattering. */
  float b;
};

__device__ inline float3 Minneart__eval(const Minneart* This,
                     const float3 &wo, const DifferentialGeometry &dg, const float3 &wi) 
{
  const float cosThetaI = clamp(dot(wi,dg.Ns));
  const float backScatter = pow(clamp(dot(wo,wi)), This->b);
  return (backScatter * cosThetaI * one_over_pi) * This->R;
}

__device__ inline float3 Minneart__sample(const Minneart* This,
                       const float3 &wo, 
                       const DifferentialGeometry &dg, 
                       Sample3f &wi, 
                       const float2 &s)  
{
  wi = cosineSampleHemisphere(s.x,s.y,dg.Ns);
  return Minneart__eval(This, wo, dg, wi.v);
}

__device__ inline void Minneart__Constructor(Minneart* This, const float3& R, const float b) 
{
  This->R = R;
  This->b = b;
}

__device__ inline Minneart make_Minneart(const float3& R, const float f) { 
  Minneart m; Minneart__Constructor(&m,R,f); return m; 
}

////////////////////////////////////////////////////////////////////////////////
//                            Velvet BRDF                                     //
////////////////////////////////////////////////////////////////////////////////

struct Velvety
{
  BRDF base;

  /*! The reflectance parameter. The vale 0 means no reflection,
   *  and 1 means full reflection. */
  float3 R;
  
  /*! The falloff of horizon scattering. 0 no falloff,
   *  and CUDA_INF means maximum falloff. */
  float f;
};

__device__ inline float3 Velvety__eval(const Velvety* This,
                    const float3 &wo, const DifferentialGeometry &dg, const float3 &wi) 
{
  const float cosThetaO = clamp(dot(wo,dg.Ns));
  const float cosThetaI = clamp(dot(wi,dg.Ns));
  const float sinThetaO = sqrt(1.0f - cosThetaO * cosThetaO);
  const float horizonScatter = pow(sinThetaO, This->f);
  return (horizonScatter * cosThetaI * one_over_pi) * This->R;
}

__device__ inline float3 Velvety__sample(const Velvety* This,
                      const float3 &wo, 
                      const DifferentialGeometry &dg, 
                      Sample3f &wi, 
                      const float2 &s)  
{
  wi = cosineSampleHemisphere(s.x,s.y,dg.Ns);
  return Velvety__eval(This, wo, dg, wi.v);
}

__device__ inline void Velvety__Constructor(Velvety* This, const float3& R, const float f) 
{
  This->R = R;
  This->f = f;
}

__device__ inline Velvety make_Velvety(const float3& R, const float f) { 
  Velvety m; Velvety__Constructor(&m,R,f); return m; 
}

////////////////////////////////////////////////////////////////////////////////
//                  Dielectric Reflection BRDF                                //
////////////////////////////////////////////////////////////////////////////////

struct DielectricReflection
{
  float eta;
};

__device__ inline float3 DielectricReflection__eval(const DielectricReflection* This, const float3 &wo, const DifferentialGeometry &dg, const float3 &wi) {
  return optix::make_float3(0.f);
}

__device__ inline float3 DielectricReflection__sample(const DielectricReflection* This, const float3 &wo, const DifferentialGeometry &dg, Sample3f &wi, const float2 &s)
{
  const float cosThetaO = clamp(dot(wo,dg.Ns));
  wi = reflect_(wo,dg.Ns,cosThetaO);
  return optix::make_float3(fresnelDielectric(cosThetaO,This->eta));
}

__device__ inline void DielectricReflection__Constructor(DielectricReflection* This,
                                              const float etai,
                                              const float etat)
{
  This->eta = etai*rcp(etat);
}

__device__ inline DielectricReflection make_DielectricReflection(const float etai, const float etat) {
  DielectricReflection v; DielectricReflection__Constructor(&v,etai,etat); return v;
}

////////////////////////////////////////////////////////////////////////////////
//                                Lambertian BRDF                             //
////////////////////////////////////////////////////////////////////////////////

struct Lambertian
{
  float3 R;
};

__device__ inline float3 Lambertian__eval(const Lambertian* This,
                              const float3 &wo, const DifferentialGeometry &dg, const float3 &wi) 
{
  return This->R * (1.0f/(float)(pi)) * clamp(dot(wi,dg.Ns));
}

__device__ inline float3 Lambertian__sample(const Lambertian* This,
                                const float3 &wo, 
                                const DifferentialGeometry &dg, 
                                Sample3f &wi, 
                                const float2 &s)  
{
  wi = cosineSampleHemisphere(s.x,s.y,dg.Ns);
  return Lambertian__eval(This, wo, dg, wi.v);
}

__device__ inline void Lambertian__Constructor(Lambertian* This, const float3& R)
{
  This->R = R;
}

__device__ inline Lambertian make_Lambertian(const float3& R) {
  Lambertian v; Lambertian__Constructor(&v,R); return v;
}


////////////////////////////////////////////////////////////////////////////////
//              Lambertian BRDF with Dielectric Layer on top                  //
////////////////////////////////////////////////////////////////////////////////

struct DielectricLayerLambertian
{
  float3 T;             //!< Transmission coefficient of dielectricum
  float etait;         //!< Relative refraction index etai/etat of both media
  float etati;         //!< relative refraction index etat/etai of both media
  Lambertian ground;   //!< the BRDF of the ground layer
};

__device__ inline float3 DielectricLayerLambertian__eval(const DielectricLayerLambertian* This,
                                             const float3 &wo, const DifferentialGeometry &dg, const float3 &wi) 
{
  const float cosThetaO = dot(wo,dg.Ns);
  const float cosThetaI = dot(wi,dg.Ns);
  if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) return optix::make_float3(0.f);

  float cosThetaO1; 
  const Sample3f wo1 = refract(wo,dg.Ns,This->etait,cosThetaO,cosThetaO1);
  float cosThetaI1; 
  const Sample3f wi1 = refract(wi,dg.Ns,This->etait,cosThetaI,cosThetaI1);
  const float Fi = 1.0f - fresnelDielectric(cosThetaI,cosThetaI1,This->etait);
  const float3 Fg = Lambertian__eval(&This->ground,neg(wo1.v),dg,neg(wi1.v));
  const float Fo = 1.0f - fresnelDielectric(cosThetaO,cosThetaO1,This->etait);
  return Fo * This->T * Fg * This->T * Fi;
}

__device__ inline float3 DielectricLayerLambertian__sample(const DielectricLayerLambertian* This,
                                               const float3 &wo, 
                                               const DifferentialGeometry &dg, 
                                               Sample3f &wi, 
                                               const float2 &s)  
{
  /*! refract ray into medium */
  float cosThetaO = dot(wo,dg.Ns);
  if (cosThetaO <= 0.0f) return optix::make_float3(0.f);
  float cosThetaO1; Sample3f wo1 = refract(wo,dg.Ns,This->etait,cosThetaO,cosThetaO1);
  
  /*! sample ground BRDF */
  Sample3f wi1 = make_Sample3f(optix::make_float3(0.f),1.f); 
  float3 Fg = Lambertian__sample(&This->ground,neg(wo1.v),dg,wi1,s);

  /*! refract ray out of medium */
  float cosThetaI1 = dot(wi1.v,dg.Ns);
  if (cosThetaI1 <= 0.0f) return optix::make_float3(0.f);
  
  float cosThetaI; 
  Sample3f wi0 = refract(neg(wi1.v),neg(dg.Ns),This->etati,cosThetaI1,cosThetaI);
  if (wi0.pdf == 0.0f) return optix::make_float3(0.f);
  
  /*! accumulate contribution of path */
  wi = make_Sample3f(wi0.v,wi1.pdf);
  float Fi = 1.0f - fresnelDielectric(cosThetaI,cosThetaI1,This->etait);
  float Fo = 1.0f - fresnelDielectric(cosThetaO,cosThetaO1,This->etait);
  return Fo * This->T * Fg * This->T * Fi;
}

__device__ inline void DielectricLayerLambertian__Constructor(DielectricLayerLambertian* This,
                                                   const float3& T, 
                                                   const float etai, 
                                                   const float etat, 
                                                   const Lambertian& ground)
{
  This->T = T;
  This->etait = etai*rcp(etat);
  This->etati = etat*rcp(etai);
  This->ground = ground;
}

__device__ inline DielectricLayerLambertian make_DielectricLayerLambertian(const float3& T, 
                                                                        const float etai, 
                                                                        const float etat, 
                                                                        const Lambertian& ground)
{
  DielectricLayerLambertian m; 
  DielectricLayerLambertian__Constructor(&m,T,etai,etat,ground);
  return m;
}

////////////////////////////////////////////////////////////////////////////////
//                          Matte Material                                    //
////////////////////////////////////////////////////////////////////////////////

__device__ void MatteMaterial__preprocess(MatteMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 MatteMaterial__eval(MatteMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) 
{
  Lambertian lambertian = make_Lambertian(optix::make_float3((float4)This->reflectance));
  return Lambertian__eval(&lambertian,wo,dg,wi);
}

__device__ float3 MatteMaterial__sample(MatteMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  Lambertian lambertian = make_Lambertian(optix::make_float3((float4)This->reflectance));
  return Lambertian__sample(&lambertian,wo,dg,wi_o,s);
}

////////////////////////////////////////////////////////////////////////////////
//                          Mirror Material                                    //
////////////////////////////////////////////////////////////////////////////////

__device__ void MirrorMaterial__preprocess(MirrorMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 MirrorMaterial__eval(MirrorMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) {
  return optix::make_float3(0.0f);
}

__device__ float3 MirrorMaterial__sample(MirrorMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  wi_o = reflect_(wo,dg.Ns);
  return optix::make_float3(This->reflectance);
}

////////////////////////////////////////////////////////////////////////////////
//                          OBJ Material                                      //
////////////////////////////////////////////////////////////////////////////////

__device__ void OBJMaterial__preprocess(OBJMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{

    float d = material->d;
    //if (material->map_d) { d *= material->map_d.get(s,t); }
    brdf.Ka = optix::make_float3(material->Ka);
    //if (material->map_Ka) { brdf.Ka *= material->map_Ka->get(dg.st); }
    brdf.Kd = d * optix::make_float3(material->Kd);  
    //if (material->map_Kd) brdf.Kd *= material->map_Kd->get(dg.st);  

#if ENABLE_TEXTURING == 1
    if (material->map_Kd) 
      {
#if ENABLE_PTEX == 1
        brdf.Kd = d * getPtexTexel3f(material->map_Kd, dg.primID, dg.v, dg.u);
#else
        brdf.Kd = getTextureTexel3f(material->map_Kd,dg.u,dg.v);	
#endif
      }
#endif

    brdf.Ks = d * optix::make_float3(material->Ks);  
    //if (material->map_Ks) brdf.Ks *= material->map_Ks->get(dg.st); 
    brdf.Ns = material->Ns;  
    //if (material->map_Ns) { brdf.Ns *= material->map_Ns.get(dg.st); }
    brdf.Kt = (1.0f-d)*optix::make_float3(material->Kt);
    brdf.Ni = material->Ni;

}

__device__ float3 OBJMaterial__eval(OBJMaterial* material, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) 
{
  float3 R = optix::make_float3(0.0f,0.0f,0.0f);
  const float Md = max(max(brdf.Kd.x,brdf.Kd.y),brdf.Kd.z);
  const float Ms = max(max(brdf.Ks.x,brdf.Ks.y),brdf.Ks.z);
  const float Mt = max(max(brdf.Kt.x,brdf.Kt.y),brdf.Kt.z);
  if (Md > 0.0f) {
    R = R + (1.0f/pi) * clamp(dot(wi,dg.Ns)) * brdf.Kd; // FIXME: +=
  }
  if (Ms > 0.0f) {
    const Sample3f refl = reflect_(wo,dg.Ns);
    if (dot(refl.v,wi) > 0.0f) 
      R = R + (brdf.Ns+2) * one_over_two_pi * pow(max(1e-10f,dot(refl.v,wi)),brdf.Ns) * clamp(dot(wi,dg.Ns)) * brdf.Ks; // FIXME: +=
  }
  if (Mt > 0.0f) {
  }
  return R;
}

__device__ float3 OBJMaterial__sample(OBJMaterial* material, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  float3 cd = optix::make_float3(0.0f); 
  Sample3f wid = make_Sample3f(optix::make_float3(0.0f),0.0f);
  if (max(max(brdf.Kd.x,brdf.Kd.y),brdf.Kd.z) > 0.0f) {
    wid = cosineSampleHemisphere(s.x,s.y,dg.Ns);
    cd = one_over_pi * clamp(dot(wid.v,dg.Ns)) * brdf.Kd;
  }

  float3 cs = optix::make_float3(0.0f); 
  Sample3f wis = make_Sample3f(optix::make_float3(0.0f),0.0f);
  if (max(max(brdf.Ks.x,brdf.Ks.y),brdf.Ks.z) > 0.0f)
  {
    const Sample3f refl = reflect_(wo,dg.Ns);
    wis = powerCosineSampleHemisphere(s.x,s.y,refl.v,brdf.Ns);
    cs = (brdf.Ns+2) * one_over_two_pi * pow(dot(refl.v,wis.v),brdf.Ns) * clamp(dot(wis.v,dg.Ns)) * brdf.Ks;
  }

  float3 ct = optix::make_float3(0.0f); 
  Sample3f wit = make_Sample3f(optix::make_float3(0.0f),0.0f);
  if (max(max(brdf.Kt.x,brdf.Kt.y),brdf.Kt.z) > 0.0f)
  {
    wit = make_Sample3f(neg(wo),1.0f);
    ct = brdf.Kt;
  }

  const float3 md = Lw*cd/wid.pdf;
  const float3 ms = Lw*cs/wis.pdf;
  const float3 mt = Lw*ct/wit.pdf;

  const float Cd = wid.pdf == 0.0f ? 0.0f : max(max(md.x,md.y),md.z);
  const float Cs = wis.pdf == 0.0f ? 0.0f : max(max(ms.x,ms.y),ms.z);
  const float Ct = wit.pdf == 0.0f ? 0.0f : max(max(mt.x,mt.y),mt.z);
  const float C  = Cd + Cs + Ct;

  if (C == 0.0f) {
    wi_o = make_Sample3f(optix::make_float3(0,0,0),0);
    return optix::make_float3(0,0,0);
  }

  const float CPd = Cd/C;
  const float CPs = Cs/C;
  const float CPt = Ct/C;

  if (s.x < CPd) {
    wi_o = make_Sample3f(wid.v,wid.pdf*CPd);
    return cd;
  } 
  else if (s.x < CPd + CPs)
  {
    wi_o = make_Sample3f(wis.v,wis.pdf*CPs);
    return cs;
  }
  else 
  {
    wi_o = make_Sample3f(wit.v,wit.pdf*CPt);
    return ct;
  }
}

////////////////////////////////////////////////////////////////////////////////
//                        Metal Material                                      //
////////////////////////////////////////////////////////////////////////////////

__device__ void MetalMaterial__preprocess(MetalMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 MetalMaterial__eval(MetalMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) 
{
  const FresnelConductor fresnel = make_FresnelConductor(optix::make_float3(This->eta),optix::make_float3(This->k));
  const PowerCosineDistribution distribution = make_PowerCosineDistribution(rcp(This->roughness));

  const float cosThetaO = dot(wo,dg.Ns);
  const float cosThetaI = dot(wi,dg.Ns);
  if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) return optix::make_float3(0.f);
  const float3 wh = normalize(wi+wo);
  const float cosThetaH = dot(wh, dg.Ns);
  const float cosTheta = dot(wi, wh); // = dot(wo, wh);
  const float3 F = eval(fresnel,cosTheta);
  const float D = eval(distribution,cosThetaH);
  const float G = min(1.0f, min(2.0f * cosThetaH * cosThetaO / cosTheta, 
                                2.0f * cosThetaH * cosThetaI / cosTheta));
  return (optix::make_float3(This->reflectance)*F) * D * G * rcp(4.0f*cosThetaO);
}

__device__ float3 MetalMaterial__sample(MetalMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  const PowerCosineDistribution distribution = make_PowerCosineDistribution(rcp(This->roughness));

  if (dot(wo,dg.Ns) <= 0.0f) return optix::make_float3(0.0f);
  sample(distribution,wo,dg.Ns,wi_o,s);
  if (dot(wi_o.v,dg.Ns) <= 0.0f) return optix::make_float3(0.0f);
  return MetalMaterial__eval(This,brdf,wo,dg,wi_o.v);
}

////////////////////////////////////////////////////////////////////////////////
//                        ReflectiveMetal Material                            //
////////////////////////////////////////////////////////////////////////////////

__device__ void ReflectiveMetalMaterial__preprocess(ReflectiveMetalMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  {
}

__device__ float3 ReflectiveMetalMaterial__eval(ReflectiveMetalMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) {
  return optix::make_float3(0.0f);
}

__device__ float3 ReflectiveMetalMaterial__sample(ReflectiveMetalMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  wi_o = reflect_(wo,dg.Ns);
  return optix::make_float3(This->reflectance) * fresnelConductor(dot(wo,dg.Ns),optix::make_float3((float4)This->eta),optix::make_float3((float4)This->k));
}

////////////////////////////////////////////////////////////////////////////////
//                        Velvet Material                                     //
////////////////////////////////////////////////////////////////////////////////

__device__ void VelvetMaterial__preprocess(VelvetMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 VelvetMaterial__eval(VelvetMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) 
{
  Minneart minneart; Minneart__Constructor(&minneart,(float3)optix::make_float3(This->reflectance),This->backScattering);
  Velvety velvety; Velvety__Constructor (&velvety,optix::make_float3((float4)This->horizonScatteringColor),This->horizonScatteringFallOff);
  return Minneart__eval(&minneart,wo,dg,wi) + Velvety__eval(&velvety,wo,dg,wi);
}

__device__ float3 VelvetMaterial__sample(VelvetMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  Minneart minneart; Minneart__Constructor(&minneart,optix::make_float3((float4)This->reflectance),This->backScattering);
  Velvety velvety; Velvety__Constructor (&velvety,optix::make_float3((float4)This->horizonScatteringColor),This->horizonScatteringFallOff);

  Sample3f wi0; float3 c0 = Minneart__sample(&minneart,wo,dg,wi0,s);
  Sample3f wi1; float3 c1 = Velvety__sample(&velvety,wo,dg,wi1,s);
  return sample_component2(c0,wi0,medium,c1,wi1,medium,Lw,wi_o,medium,s.x);
}

////////////////////////////////////////////////////////////////////////////////
//                          Dielectric Material                               //
////////////////////////////////////////////////////////////////////////////////

__device__ void DielectricMaterial__preprocess(DielectricMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 DielectricMaterial__eval(DielectricMaterial* material, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) {
  return optix::make_float3(0.0f);
}

__device__ float3 DielectricMaterial__sample(DielectricMaterial* material, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  float eta = 0.0f;
  Medium mediumOutside = make_Medium(optix::make_float3((float4)material->transmissionOutside),material->etaOutside);
  Medium mediumInside  = make_Medium(optix::make_float3((float4)material->transmissionInside ),material->etaInside );
  Medium mediumFront, mediumBack;
  if (eq(medium,mediumInside)) {
    eta = material->etaInside/material->etaOutside;
    mediumFront = mediumInside;
    mediumBack = mediumOutside;
  }
  else {
    eta = material->etaOutside/material->etaInside;
    mediumFront = mediumOutside;
    mediumBack = mediumInside;
  }

  float cosThetaO = clamp(dot(wo,dg.Ns));
  float cosThetaI; Sample3f wit = refract(wo,dg.Ns,eta,cosThetaO,cosThetaI);
  Sample3f wis = reflect_(wo,dg.Ns);
  float R = fresnelDielectric(cosThetaO,cosThetaI,eta);
  float3 cs = optix::make_float3(R);
  float3 ct = optix::make_float3(1.0f-R);
  return sample_component2(cs,wis,mediumFront,ct,wit,mediumBack,Lw,wi_o,medium,s.x);
}

////////////////////////////////////////////////////////////////////////////////
//                          ThinDielectric Material                               //
////////////////////////////////////////////////////////////////////////////////

__device__ void ThinDielectricMaterial__preprocess(ThinDielectricMaterial* This, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 ThinDielectricMaterial__eval(ThinDielectricMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) {
  return optix::make_float3(0.0f);
}

__device__ float3 ThinDielectricMaterial__sample(ThinDielectricMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  float cosThetaO = clamp(dot(wo,dg.Ns));
  if (cosThetaO <= 0.0f) return optix::make_float3(0.0f);
  float R = fresnelDielectric(cosThetaO,rcp(This->eta));
  Sample3f wit = make_Sample3f(neg(wo),1.0f);
  Sample3f wis = reflect_(wo,dg.Ns);
  float3 ct = exp(optix::make_float3(This->transmission)*rcp(cosThetaO))*optix::make_float3(1.0f-R);
  float3 cs = optix::make_float3(R);
  return sample_component2(cs,wis,medium,ct,wit,medium,Lw,wi_o,medium,s.x);
}

////////////////////////////////////////////////////////////////////////////////
//                     MetallicPaint Material                                 //
////////////////////////////////////////////////////////////////////////////////

__device__ void MetallicPaintMaterial__preprocess(MetallicPaintMaterial* material, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
}

__device__ float3 MetallicPaintMaterial__eval(MetallicPaintMaterial* This, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi) 
{
  DielectricReflection reflection; DielectricReflection__Constructor(&reflection, 1.0f, This->eta);
  DielectricLayerLambertian lambertian; DielectricLayerLambertian__Constructor(&lambertian, optix::make_float3((float)1.0f), 1.0f, This->eta, make_Lambertian(optix::make_float3((float4)This->shadeColor)));
  return DielectricReflection__eval(&reflection,wo,dg,wi) + DielectricLayerLambertian__eval(&lambertian,wo,dg,wi);
}

__device__ float3 MetallicPaintMaterial__sample(MetallicPaintMaterial* This, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{
  DielectricReflection reflection; DielectricReflection__Constructor(&reflection, 1.0f, This->eta);
  DielectricLayerLambertian lambertian; DielectricLayerLambertian__Constructor(&lambertian, optix::make_float3((float)1.0f), 1.0f, This->eta, make_Lambertian(optix::make_float3((float4)This->shadeColor)));
  Sample3f wi0; float3 c0 = DielectricReflection__sample(&reflection,wo,dg,wi0,s);
  Sample3f wi1; float3 c1 = DielectricLayerLambertian__sample(&lambertian,wo,dg,wi1,s);
  return sample_component2(c0,wi0,medium,c1,wi1,medium,Lw,wi_o,medium,s.x);
}

////////////////////////////////////////////////////////////////////////////////
//                              Material                                      //
////////////////////////////////////////////////////////////////////////////////

__device__ inline void Material__preprocess(ISPCMaterial* materials, int materialID, int numMaterials, BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const Medium& medium)  
{
  
  {
    
  ISPCMaterial* material = &materials[materialID];

  switch (material->ty) {
  case MATERIAL_OBJ  : OBJMaterial__preprocess  ((OBJMaterial*)  material,brdf,wo,dg,medium); break;
  case MATERIAL_METAL: MetalMaterial__preprocess((MetalMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_REFLECTIVE_METAL: ReflectiveMetalMaterial__preprocess((ReflectiveMetalMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_VELVET: VelvetMaterial__preprocess((VelvetMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_DIELECTRIC: DielectricMaterial__preprocess((DielectricMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_METALLIC_PAINT: MetallicPaintMaterial__preprocess((MetallicPaintMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_MATTE: MatteMaterial__preprocess((MatteMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_MIRROR: MirrorMaterial__preprocess((MirrorMaterial*)material,brdf,wo,dg,medium); break;
  case MATERIAL_THIN_DIELECTRIC: ThinDielectricMaterial__preprocess((ThinDielectricMaterial*)material,brdf,wo,dg,medium); break;
  default: break;
  }
  }
}

__device__ inline float3 Material__eval(ISPCMaterial* materials, int materialID, int numMaterials, const BRDF& brdf, const float3& wo, const DifferentialGeometry& dg, const float3& wi)
{
  float3 c = optix::make_float3(0.0f);
  
  {
    
  ISPCMaterial* material = &materials[materialID];
  switch (material->ty) {
  case MATERIAL_OBJ  : c = OBJMaterial__eval  ((OBJMaterial*)  material, brdf, wo, dg, wi); break;
  case MATERIAL_METAL: c = MetalMaterial__eval((MetalMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_REFLECTIVE_METAL: c = ReflectiveMetalMaterial__eval((ReflectiveMetalMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_VELVET: c = VelvetMaterial__eval((VelvetMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_DIELECTRIC: c = DielectricMaterial__eval((DielectricMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_METALLIC_PAINT: c = MetallicPaintMaterial__eval((MetallicPaintMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_MATTE: c = MatteMaterial__eval((MatteMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_MIRROR: c = MirrorMaterial__eval((MirrorMaterial*)material, brdf, wo, dg, wi); break;
  case MATERIAL_THIN_DIELECTRIC: c = ThinDielectricMaterial__eval((ThinDielectricMaterial*)material, brdf, wo, dg, wi); break;
  default: c = optix::make_float3(0.0f); 
  }
  }
  return c;
}

__device__ inline float3 Material__sample(ISPCMaterial* materials, int materialID, int numMaterials, const BRDF& brdf, const float3& Lw, const float3& wo, const DifferentialGeometry& dg, Sample3f& wi_o, Medium& medium, const float2& s)  
{  
  float3 c = optix::make_float3(0.0f);
  
  {
    
  ISPCMaterial* material = &materials[materialID];
  switch (material->ty) {
  case MATERIAL_OBJ  : c = OBJMaterial__sample  ((OBJMaterial*)  material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_METAL: c = MetalMaterial__sample((MetalMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_REFLECTIVE_METAL: c = ReflectiveMetalMaterial__sample((ReflectiveMetalMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_VELVET: c = VelvetMaterial__sample((VelvetMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_DIELECTRIC: c = DielectricMaterial__sample((DielectricMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_METALLIC_PAINT: c = MetallicPaintMaterial__sample((MetallicPaintMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_MATTE: c = MatteMaterial__sample((MatteMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_MIRROR: c = MirrorMaterial__sample((MirrorMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  case MATERIAL_THIN_DIELECTRIC: c = ThinDielectricMaterial__sample((ThinDielectricMaterial*)material, brdf, Lw, wo, dg, wi_o, medium, s); break;
  default: c = optix::make_float3(0.0f); 
  }
  }
  return c;
}

#if 0

////////////////////////////////////////////////////////////////////////////////
//                               Scene                                        //
////////////////////////////////////////////////////////////////////////////////

/* scene data */
//extern ISPCScene* g_ispc_scene;
//RTCScene g_scene = NULL;
void** geomID_to_mesh = NULL;
int* geomID_to_type = NULL;

/* render function to use */
//renderPixelFunc renderPixel;

/* occlusion filter function */
#if ENABLE_OCCLUSION_FILTER == 1
void occlusionFilterReject(void* ptr, optix::Ray& ray) {
  hit.primID = -1;
}
#endif

/* error reporting function */
void error_handler(const RTCError code, const int8* str)
{
  print("Embree: ");
  switch (code) {
  case RTC_UNKNOWN_ERROR    : print("RTC_UNKNOWN_ERROR"); break;
  case RTC_INVALID_ARGUMENT : print("RTC_INVALID_ARGUMENT"); break;
  case RTC_INVALID_OPERATION: print("RTC_INVALID_OPERATION"); break;
  case RTC_OUT_OF_MEMORY    : print("RTC_OUT_OF_MEMORY"); break;
  case RTC_UNSUPPORTED_CPU  : print("RTC_UNSUPPORTED_CPU"); break;
  case RTC_CANCELLED        : print("RTC_CANCELLED"); break;
  default                   : print("invalid error code"); break;
  }
  if (str) { 
    print(" ("); 
    while (*str) putchar(*str++); 
    print(")\n"); 
  }
  abort();
} // error handler

/* accumulation buffer */
float4* g_accu = NULL;
unsigned int g_accu_width = 0;
unsigned int g_accu_height = 0;
__device__ unsigned int g_accu_count = 0;
float3 g_accu_vx;
float3 g_accu_vy;
float3 g_accu_vz;
float3 g_accu_p;
extern bool g_changed;

/* called by the C++ code for initialization */
export void device_init (int8* cfg)
{
  /* initialize last seen camera */
  g_accu_vx = optix::make_float3(0.0f);
  g_accu_vy = optix::make_float3(0.0f);
  g_accu_vz = optix::make_float3(0.0f);
  g_accu_p  = optix::make_float3(0.0f);

  /* initialize ray tracing core */
  rtcInit(cfg);

  /* set error handler */
  rtcSetErrorFunction(error_handler);

  /* set start render mode */
  renderPixel = renderPixelStandard;
  //  renderPixel = renderPixelEyeLight;

} // device_init


#if ENABLE_DISPLACEMENTS
void displacementFunction(void* ptr, unsigned int geomID, int unsigned primID, 
                      const float* u,      /*!< u coordinates (source) */
                      const float* v,      /*!< v coordinates (source) */
                      const float* nx,     /*!< x coordinates of normal at point to displace (source) */
                      const float* ny,     /*!< y coordinates of normal at point to displace (source) */
                      const float* nz,     /*!< z coordinates of normal at point to displace (source) */
                      float* px,           /*!< x coordinates of points to displace (source and target) */
                      float* py,           /*!< y coordinates of points to displace (source and target) */
                      float* pz,           /*!< z coordinates of points to displace (source and target) */
                      size_t N)
{
  ISPCSubdivMesh* mesh = (ISPCSubdivMesh*)geomID_to_mesh[geomID];
  int materialID = mesh->materialID;
  int numMaterials = 0;
  OBJMaterial* material = (OBJMaterial*)&materials[materialID];
  if (material->map_Displ)
    for (size_t i=0;i<N;i++) 
      {
	const float displ = getPtexTexel1f(material->map_Displ, primID, v[i], u[i]);
	px[i] += nx[i] * displ;
	py[i] += ny[i] * displ;
	pz[i] += nz[i] * displ;
      }
}
#endif

void convertTriangleMeshes(ISPCScene* scene_in, RTCScene scene_out, size_t numGeometries)
{
  /* add all meshes to the scene */
  for (int i=0; i<scene_in->numMeshes; i++)
  {

    /* get ith mesh */
    ISPCMesh* mesh = scene_in->meshes[i];

    /* create a triangle mesh */
    unsigned int geomID = rtcNewTriangleMesh (scene_out, RTC_GEOMETRY_STATIC, mesh->numTriangles, mesh->numVertices);
    assert(geomID < numGeometries);
    geomID_to_mesh[geomID] = mesh;
    geomID_to_type[geomID] = 0;
    
    /* set vertices */
    Vertex* vertices = (Vertex*) rtcMapBuffer(scene_out,geomID,RTC_VERTEX_BUFFER); 
    for (int j=0; j<mesh->numVertices; j++) {
      vertices[j].x = mesh->positions[j].x;
      vertices[j].y = mesh->positions[j].y;
      vertices[j].z = mesh->positions[j].z;
    }
    rtcUnmapBuffer(scene_out,geomID,RTC_VERTEX_BUFFER); 

    /* set triangles */
    Triangle* triangles = (Triangle*) rtcMapBuffer(scene_out,geomID,RTC_INDEX_BUFFER);
    for (int j=0; j<mesh->numTriangles; j++) {
      triangles[j].v0 = mesh->triangles[j].v0;
      triangles[j].v1 = mesh->triangles[j].v1;
      triangles[j].v2 = mesh->triangles[j].v2;
    }
    rtcUnmapBuffer(scene_out,geomID,RTC_INDEX_BUFFER);

    bool allOpaque = true;
    bool allTransparent = true;
    for (size_t j=0; j<mesh->numTriangles; j++) {
      ISPCTriangle triangle = mesh->triangles[j];
      if (materials[triangle.materialID].ty == MATERIAL_DIELECTRIC ||
	  materials[triangle.materialID].ty == MATERIAL_THIN_DIELECTRIC)
	allOpaque = false;
      else 
	allTransparent = false;
    }

#if ENABLE_OCCLUSION_FILTER == 1
    if (allTransparent)
      rtcSetOcclusionFilterFunction(scene_out,geomID,(RTCFilterFuncVarying)&occlusionFilterReject);
#endif

  }
}

__device__ inline float updateEdgeLevel( ISPCSubdivMesh* mesh, const float4& cam_pos, const size_t e0, const size_t e1)
{
  const float4 v0 = mesh->positions[mesh->position_indices[e0]];
  const float4 v1 = mesh->positions[mesh->position_indices[e1]];
  const float4 edge = v1-v0;
  const float4 P = 0.5f*(v1+v0);
  const float4 dist = cam_pos - P;
  return max(min(LEVEL_FACTOR*(0.5f*length(edge)/length(dist)),MAX_EDGE_LEVEL),MIN_EDGE_LEVEL);
}

void updateEdgeLevelBuffer( ISPCSubdivMesh* mesh, const float4& cam_pos, size_t startID, size_t endID )
{
  for (size_t f=startID; f<endID;f++) {
       int e = mesh->face_offsets[f];
       int N = mesh->verticesPerFace[f];
       if (N == 4) /* fast path for quads */
         for (size_t i=0; i<4; i++) 
           mesh->subdivlevel[e+i] =  updateEdgeLevel(mesh,cam_pos,e+(i+0),e+(i+1)%4);
       else if (N == 3) /* fast path for triangles */
         for (size_t i=0; i<3; i++) 
           mesh->subdivlevel[e+i] =  updateEdgeLevel(mesh,cam_pos,e+(i+0),e+(i+1)%3);
       else /* fast path for general polygons */
        for (size_t i=0; i<N; i++) 
           mesh->subdivlevel[e+i] =  updateEdgeLevel(mesh,cam_pos,e+(i+0),e+(i+1)%N);              
 }
}

#if defined(ISPC)
task void updateEdgeLevelBufferTask( ISPCSubdivMesh* mesh, const float4& cam_pos )
{
  const size_t size = mesh->numFaces;
  const size_t startID = ((taskIndex+0)*size)/taskCount;
  const size_t endID   = ((taskIndex+1)*size)/taskCount;
  updateEdgeLevelBuffer(mesh,cam_pos,startID,endID);
}
#endif

void updateKeyFrame(ISPCScene* scene_in)
{
  for (size_t g=0; g<scene_in->numSubdivMeshes; g++)
  {
    ISPCSubdivMesh* mesh = g_ispc_scene->subdiv[g];
    unsigned int geomID = mesh->geomID;

    if (g_ispc_scene->subdivMeshKeyFrames)
      {
	ISPCSubdivMeshKeyFrame *keyframe      = g_ispc_scene->subdivMeshKeyFrames[keyframeID];
	ISPCSubdivMesh         *keyframe_mesh = keyframe->subdiv[g];
	rtcSetBuffer(g_scene, geomID, RTC_VERTEX_BUFFER, keyframe_mesh->positions, 0, sizeof(float4  ));
	rtcUpdateBuffer(g_scene,geomID,RTC_VERTEX_BUFFER);    
      }
  }

  keyframeID++;
  if (keyframeID >= g_ispc_scene->numSubdivMeshKeyFrames)
    keyframeID = 0;

}


void updateEdgeLevels(ISPCScene* scene_in, const float4& cam_pos)
{
  for (size_t g=0; g<scene_in->numSubdivMeshes; g++)
  {
    ISPCSubdivMesh* mesh = g_ispc_scene->subdiv[g];
    unsigned int geomID = mesh->geomID;
#if defined(ISPC)
      launch[ getNumHWThreads() ] updateEdgeLevelBufferTask(mesh,cam_pos); sync;	           
#else
      updateEdgeLevelBuffer(mesh,cam_pos,0,mesh->numFaces);
#endif
   rtcUpdateBuffer(g_scene,geomID,RTC_LEVEL_BUFFER);    
  }
}



void convertSubdivMeshes(ISPCScene* scene_in, RTCScene scene_out, size_t numGeometries, const float4& cam_pos)
{
  for (size_t i=0; i<g_ispc_scene->numSubdivMeshes; i++)
  {
    ISPCSubdivMesh* mesh = g_ispc_scene->subdiv[i];
    unsigned int geomID = rtcNewSubdivisionMesh(scene_out, RTC_GEOMETRY_DYNAMIC, mesh->numFaces, mesh->numEdges, mesh->numVertices, 
						mesh->numEdgeCreases, mesh->numVertexCreases, mesh->numHoles);
    mesh->geomID = geomID;												
    assert(geomID < numGeometries);
    geomID_to_mesh[geomID] = mesh;
    geomID_to_type[geomID] = 1; //2

    for (size_t i=0; i<mesh->numEdges; i++) mesh->subdivlevel[i] = FIXED_EDGE_TESSELLATION_VALUE;
    rtcSetBuffer(scene_out, geomID, RTC_VERTEX_BUFFER, mesh->positions, 0, sizeof(float4  ));
    rtcSetBuffer(scene_out, geomID, RTC_LEVEL_BUFFER,  mesh->subdivlevel, 0, sizeof(float));
    rtcSetBuffer(scene_out, geomID, RTC_INDEX_BUFFER,  mesh->position_indices  , 0, sizeof(unsigned int));
    rtcSetBuffer(scene_out, geomID, RTC_FACE_BUFFER,   mesh->verticesPerFace, 0, sizeof(unsigned int));
    rtcSetBuffer(scene_out, geomID, RTC_HOLE_BUFFER,   mesh->holes, 0, sizeof(unsigned int));
    rtcSetBuffer(scene_out, geomID, RTC_EDGE_CREASE_INDEX_BUFFER,    mesh->edge_creases,          0, 2*sizeof(unsigned int));
    rtcSetBuffer(scene_out, geomID, RTC_EDGE_CREASE_WEIGHT_BUFFER,   mesh->edge_crease_weights,   0, sizeof(float));
    rtcSetBuffer(scene_out, geomID, RTC_VERTEX_CREASE_INDEX_BUFFER,  mesh->vertex_creases,        0, sizeof(unsigned int));
    rtcSetBuffer(scene_out, geomID, RTC_VERTEX_CREASE_WEIGHT_BUFFER, mesh->vertex_crease_weights, 0, sizeof(float));
#if ENABLE_DISPLACEMENTS == 1
      rtcSetDisplacementFunction(scene_out,geomID,(RTCDisplacementFunc)&displacementFunction,NULL);
#endif
  }
}      




typedef void* void_ptr;

RTCScene convertScene(ISPCScene* scene_in,const float4& cam_org)
{  
  size_t numGeometries = scene_in->numMeshes + scene_in->numSubdivMeshes;

  geomID_to_mesh = new void_ptr[numGeometries];
  geomID_to_type = new int[numGeometries];

  /* create scene */
  int scene_flags = RTC_SCENE_STATIC | RTC_SCENE_INCOHERENT;

  if (g_subdiv_mode)   
    scene_flags = RTC_SCENE_DYNAMIC | RTC_SCENE_INCOHERENT | RTC_SCENE_ROBUST;

  RTCScene scene_out = rtcNewScene((RTCSceneFlags)scene_flags, RTC_INTERSECT_UNIFORM | RTC_INTERSECT_VARYING);
  convertTriangleMeshes(scene_in,scene_out,numGeometries);
  convertSubdivMeshes(scene_in,scene_out,numGeometries,cam_org);


  /* commit changes to scene */
  //progressStart();
  //rtcSetProgressMonitorFunction(scene_out,progressMonitor,NULL);
  rtcCommit (scene_out);
  //rtcSetProgressMonitorFunction(scene_out,NULL,NULL);
  //progressEnd();

  return scene_out;
} // convertScene

#endif

/* for details about this random number generator see: P. L'Ecuyer,
   "Maximally Equidistributed Combined Tausworthe Generators",
   Mathematics of Computation, 65, 213 (1996), 203--213:
   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme.ps */

struct rand_state {
  unsigned int s1, s2, s3;
};

__device__ inline unsigned int irand(rand_state& state)
{
  state.s1 = ((state.s1 & 4294967294U) << 12U) ^ (((state.s1<<13U)^state.s1)>>19U);
  state.s2 = ((state.s2 & 4294967288U) <<  4U) ^ (((state.s2<< 2U)^state.s2)>>25U);
  state.s3 = ((state.s3 & 4294967280U) << 17U) ^ (((state.s3<< 3U)^state.s3)>>11U);
  return state.s1 ^ state.s2 ^ state.s3;
}

__device__ inline void init_rand(rand_state& state, unsigned int x, unsigned int y, unsigned int z)
{
  state.s1 = x >=   2 ? x : x +   2;
  state.s2 = y >=   8 ? y : y +   8;
  state.s3 = z >=  16 ? z : z +  16;
  for (int i=0; i<10; i++) irand(state);
}

__device__ inline float frand(rand_state& state) {
  return irand(state)*2.3283064365386963e-10f;
}

__device__ inline float2 frand2(rand_state& state) 
{
  float x = frand(state);
  float y = frand(state);
  return make_float2(x,y);
}

__device__ inline float3 face_forward(const float3& dir, const float3& _Ng) {
  const float3 Ng = _Ng;
  return dot(dir,Ng) < 0.0f ? Ng : neg(Ng);
}

#if 0
__device__ inline float3 interpolate_normal(optix::Ray& ray)
{
#if 0 // FIXME: pointer gather not implemented on ISPC for Xeon Phi
  ISPCMesh* mesh = g_ispc_scene->meshes[hit.primID];
  ISPCTriangle* tri = &mesh->triangles[hit.primID];

  /* load material ID */
  int materialID = tri->materialID;

  /* interpolate shading normal */
  if (mesh->normals) {
    float3 n0 = optix::make_float3(mesh->normals[tri->v0]);
    float3 n1 = optix::make_float3(mesh->normals[tri->v1]);
    float3 n2 = optix::make_float3(mesh->normals[tri->v2]);
    float u = hit.u, v = hit.v, w = 1.0f-hit.u-hit.v;
    return normalize(w*n0 + u*n1 + v*n2);
  } else {
    return normalize(hit.Ng);
  }

#else

  float3 Ns = optix::make_float3(0.0f);
  int materialID = 0;
  foreach_unique (geomID in hit.primID) 
  {
    if (geomID >= 0 && geomID < g_ispc_scene->numMeshes)  { // FIXME: workaround for ISPC bug

    ISPCMesh* mesh = g_ispc_scene->meshes[geomID];
    
    foreach_unique (primID in hit.primID) 
    {
      ISPCTriangle* tri = &mesh->triangles[primID];
      
      /* load material ID */
      materialID = tri->materialID;

      /* interpolate shading normal */
      if (mesh->normals) {
        float3 n0 = optix::make_float3(mesh->normals[tri->v0]);
        float3 n1 = optix::make_float3(mesh->normals[tri->v1]);
        float3 n2 = optix::make_float3(mesh->normals[tri->v2]);
        float u = hit.u, v = hit.v, w = 1.0f-hit.u-hit.v;
        Ns = w*n0 + u*n1 + v*n2;
      } else {
        Ns = normalize(hit.Ng);
      }
    }
    }
  }
  return normalize(Ns);
#endif
}
#endif

#if 0
#if 0 // FIXME: pointer gather not implemented in ISPC for Xeon Phi
__device__ inline int getMaterialID(const optix::Ray& ray, DifferentialGeometry& dg)
{
  int materialID = 0;
  if (geomID_to_type[hit.primID] == 0)
    materialID = ((ISPCMesh*) geomID_to_mesh[hit.primID])->triangles[hit.primID].materialID; 
  else if (geomID_to_type[hit.primID] == 1)       
  {                      
    materialID = ((ISPCSubdivMesh*) geomID_to_mesh[hit.primID])->materialID; 
#if ENABLE_TEXTURING == 1
    const float2 st = getTextureCoordinatesSubdivMesh((ISPCSubdivMesh*) geomID_to_mesh[hit.primID],hit.primID,hit.u,hit.v);
    dg.u = st.x;
    dg.v = st.y;
#endif
  }
  else
    materialID = ((ISPCMesh*) geomID_to_mesh[hit.primID])->meshMaterialID; 
  
  return materialID;
}
#else 
__device__ inline int getMaterialID(const optix::Ray& ray, DifferentialGeometry dg)
{
  int materialID = 0;
  foreach_unique (geomID in hit.primID) {
    
    if (geomID >= 0 && geomID < g_ispc_scene->numMeshes+g_ispc_scene->numSubdivMeshes) { // FIXME: workaround for ISPC bug
      if (geomID_to_type[geomID] == 0) 
	materialID = ((ISPCMesh*) geomID_to_mesh[geomID])->triangles[hit.primID].materialID; 
      else if (geomID_to_type[geomID] == 1)                
      {             
	materialID = ((ISPCSubdivMesh*) geomID_to_mesh[geomID])->materialID; 
#if ENABLE_TEXTURE_COORDINATES == 1
	const float2 st = getTextureCoordinatesSubdivMesh((ISPCSubdivMesh*) geomID_to_mesh[geomID],hit.primID,hit.u,hit.v);
	dg.u = st.x;
	dg.v = st.y;
#endif
      }
      else 
	materialID = ((ISPCMesh*) geomID_to_mesh[geomID])->meshMaterialID;         
    }
  }
  return materialID;
}
#endif
#endif

__device__ float3 renderPixelFunction(float x, float y, rand_state& state, const float3& vx, const float3& vy, const float3& vz, const float3& p)
{

  /* radiance accumulator and weight */
  float3 L = optix::make_float3(0.0f);
  float3 Lw = optix::make_float3(1.0f);
  Medium medium = make_Medium_Vacuum();

  /* initialize ray */
  Hit hit; optix::Ray ray = make_Ray(p,normalize(x*vx + y*vy + vz),0.0f,CUDA_INF);

  /* iterative path tracer loop */
  for (int i=0; i<MAX_PATH_LENGTH; i++)
  {
    /* terminate if contribution too low */
    if (max(Lw.x,max(Lw.y,Lw.z)) < 0.01f)
      break;

    /* intersect ray with scene */ 
    COUNT_RAYS(numRays++); hit.u = hit.v = hit.t = 0.0f; hit.Ng = optix::make_float3(0,0,0); hit.primID = -1; rtTrace(g_scene, ray, hit);
    const float3 wo = neg(ray.direction);
    
    /* invoke environment lights if nothing hit */
    if (hit.primID == -1) 
    {
#if 1
      /* iterate over all ambient lights */
      for (size_t i=0; i<numAmbientLights; i++)
        L = L + Lw*AmbientLight__eval(ambientLights[i],ray.direction); // FIXME: +=
#endif

#if 0
      /* iterate over all distant lights */
      for (size_t i=0; i<numDistantLights; i++)
        L = L + Lw*DistantLight__eval(*(ISPCDistantLight*)&distantLights[i],ray.direction); // FIXME: +=
#endif
      break;
    }

    /* compute differential geometry */
    DifferentialGeometry dg;
    dg.geomID = hit.primID;
    dg.primID = hit.primID;
    dg.u = hit.u;
    dg.v = hit.v;
    dg.P  = ray.origin+hit.t*ray.direction;
    dg.Ng = face_forward(ray.direction,normalize(hit.Ng));
    //float3 _Ns = interpolate_normal(ray);
    float3 _Ns = normalize(hit.Ng);
    dg.Ns = face_forward(ray.direction,_Ns);

    /* shade all rays that hit something */
    int materialID = materialID = trimaterialID[hit.primID];

    /*! Compute  simple volumetric effect. */
    float3 c = optix::make_float3(1.0f);
    const float3 transmission = medium.transmission;
    if (ne(transmission,optix::make_float3(1.0f)))
      c = c * pow(transmission,hit.t);
    
    /* calculate BRDF */ // FIXME: avoid gathers
    BRDF brdf;
    int numMaterials = 0;
    //ISPCMaterial* material = &materials[materialID];
    ISPCMaterial* material_array = &materials[0];


    Material__preprocess(material_array,materialID,numMaterials,brdf,wo,dg,medium);


    /* sample BRDF at hit point */
    Sample3f wi1;
    c = c * Material__sample(material_array,materialID,numMaterials,brdf,Lw, wo, dg, wi1, medium, frand2(state));

    
    /* iterate over ambient lights */
    for (size_t i=0; i<numAmbientLights; i++)
    {
#if 1
      float3 L0 = optix::make_float3(0.0f);
      Sample3f wi0; float tMax0;
      float3 Ll0 = AmbientLight__sample(ambientLights[i],dg,wi0,tMax0,frand2(state));

      if (wi0.pdf > 0.0f) {
        optix::Ray shadow = make_Ray(dg.P,wi0.v,0.001f,tMax0);
        COUNT_RAYS(numRays++); Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);
        if (shit.primID == -1) {
          L0 = Ll0/wi0.pdf*Material__eval(material_array,materialID,numMaterials,brdf,wo,dg,wi0.v);
        }

        L = L + Lw*L0;
      }
#endif

#if 0
      float3 L1 = optix::make_float3(0.0f);
      float3 Ll1 = AmbientLight__eval(ambientLights[i],wi1.v);
      if (wi1.pdf > 0.0f) {
        optix::Ray shadow = make_Ray(dg.P,wi1.v,0.001f,CUDA_INF);
        COUNT_RAYS(numRays++); Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);
        if (shit.primID == -1) {
          L1 = Ll1/wi1.pdf*c;
        }
        L = L + Lw*L1;
      }
#endif

#if 0
      float s = wi0.pdf*wi0.pdf + wi1.pdf*wi1.pdf;
      if (s > 0) {
        float w0 = 0;
        float w1 = 1;
        //float w0 = wi0.pdf*wi0.pdf/s;
        //float w1 = wi1.pdf*wi1.pdf/s;
        L = L + Lw*(w0*L0+w1*L1);
      }
#endif
    }
    Sample3f wi; float tMax;

    /* iterate over point lights */
    for (size_t i=0; i<numPointLights; i++)
    {
      float3 Ll = PointLight__sample(pointLights[i],dg,wi,tMax,frand2(state));
      if (wi.pdf <= 0.0f) continue;
      optix::Ray shadow = make_Ray(dg.P,wi.v,0.001f,tMax);
      COUNT_RAYS(numRays++); Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);
      if (shit.primID != -1) continue;
      L = L + Lw*Ll/wi.pdf*Material__eval(material_array,materialID,numMaterials,brdf,wo,dg,wi.v); // FIXME: +=
    }

    /* iterate over directional lights */
    for (size_t i=0; i<numDirectionalLights; i++)
    {
      float3 Ll = DirectionalLight__sample(*(ISPCDirectionalLight*)&dirLights[i],dg,wi,tMax,frand2(state));
      if (wi.pdf <= 0.0f) continue;
      optix::Ray shadow = make_Ray(dg.P,wi.v,0.001f,tMax);
      COUNT_RAYS(numRays++); Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);
      if (shit.primID != -1) continue;
      L = L + Lw*Ll/wi.pdf*Material__eval(material_array,materialID,numMaterials,brdf,wo,dg,wi.v); // FIXME: +=
    }

    /* iterate over distant lights */
    for (size_t i=0; i<numDistantLights; i++)
    {
      float3 Ll = DistantLight__sample(*(ISPCDistantLight*)&distantLights[i],dg,wi,tMax,frand2(state));

      if (wi.pdf <= 0.0f) continue;
      optix::Ray shadow = make_Ray(dg.P,wi.v,0.001f,tMax);
      COUNT_RAYS(numRays++); Hit shit; shit.u = shit.v = shit.t = 0.0f; shit.Ng = optix::make_float3(0,0,0); shit.primID = -1; rtTrace(g_scene, shadow, shit);
      if (shit.primID != -1) continue;
      L = L + Lw*Ll/wi.pdf*Material__eval(material_array,materialID,numMaterials,brdf,wo,dg,wi.v); // FIXME: +=
    }
    if (wi1.pdf <= 1E-4f /* 0.0f */) break;
    Lw = Lw*c/wi1.pdf; // FIXME: *=

    /* setup secondary ray */
    ray = make_Ray(dg.P,normalize(wi1.v),0.001f,CUDA_INF);
  }
  return L;
}

/* task that renders a single screen tile */
__device__ float3 renderPixelStandard(float x, float y, const float3& vx, const float3& vy, const float3& vz, const float3& p)
{
  rand_state state;

  float3 L = optix::make_float3(0.0f,0.0f,0.0f);

  for (int i=0; i<SAMPLES_PER_PIXEL; i++) {

  init_rand(state,
            253*x+35*y+152*g_accu_count+54,
            1253*x+345*y+1452*g_accu_count+564,
            10253*x+3435*y+52*g_accu_count+13+i*1793);

  L = L + renderPixelFunction(x,y,state,vx,vy,vz,p); 
  }
  L = L*(1.0f/SAMPLES_PER_PIXEL);
  return L;
}

#if 0
  
/* task that renders a single screen tile */
task void renderTile(int* pixels,
                     const int width,
                     const int height, 
                     const float time,
                     const float3& vx, 
                     const float3& vy, 
                     const float3& vz, 
                     const float3& p,
                     const int numTilesX, 
                     const int numTilesY)
{
  const int tileY = taskIndex / numTilesX;
  const int tileX = taskIndex - tileY * numTilesX;
  const int x0 = tileX * TILE_SIZE_X;
  const int x1 = min(x0+TILE_SIZE_X,width);
  const int y0 = tileY * TILE_SIZE_Y;
  const int y1 = min(y0+TILE_SIZE_Y,height);

  foreach_tiled (y = y0 ... y1, x = x0 ... x1)
  {
    /* calculate pixel color */
    float3 color = renderPixel(x,y,vx,vy,vz,p);

    /* write color to framebuffer */
    float4* dst = &g_accu[y*width+x];
    *dst = *dst + make_float4(color.x,color.y,color.z,1.0f); // FIXME: use += operator
    float f = rcp(max(0.001f,dst->w));
    unsigned int r = (unsigned int) (255.0f * clamp(dst->x*f,0.0f,1.0f));
    unsigned int g = (unsigned int) (255.0f * clamp(dst->y*f,0.0f,1.0f));
    unsigned int b = (unsigned int) (255.0f * clamp(dst->z*f,0.0f,1.0f));
    pixels[y*width+x] = (b << 16) + (g << 8) + r;
  }
} // renderTile

/* called by the C++ code to render */
export void device_render (int* pixels,
                           const int width,
                           const int height, 
                           const float time,
                           const float3& vx, 
                           const float3& vy, 
                           const float3& vz, 
                           const float3& p)
{
  COUNT_RAYS(numRays = 0);
  float4 cam_org = make_float4(p.x,p.y,p.z);

  /* create scene */
  if (g_scene == NULL)
   {
     if (g_ispc_scene->numSubdivMeshes > 0)
       g_subdiv_mode = true;

     g_scene = convertScene(g_ispc_scene,cam_org);

#if !defined(FORCE_FIXED_EDGE_TESSELLATION)
    if (g_subdiv_mode)
      updateEdgeLevels(g_ispc_scene, cam_org);
#endif

   }

  /* create accumulator */
  if (g_accu_width != width || g_accu_height != height) {
    delete[] g_accu;
    g_accu = new float4[width*height];
    g_accu_width = width;
    g_accu_height = height;
    memset(g_accu,0,width*height*sizeof(float4));
  }

  /* reset accumulator */
  bool camera_changed = g_changed; g_changed = false;
  camera_changed |= ne(g_accu_vx,vx); g_accu_vx = vx; // FIXME: use != operator
  camera_changed |= ne(g_accu_vy,vy); g_accu_vy = vy; // FIXME: use != operator
  camera_changed |= ne(g_accu_vz,vz); g_accu_vz = vz; // FIXME: use != operator
  camera_changed |= ne(g_accu_p,  p); g_accu_p  = p;  // FIXME: use != operator

  if (g_ispc_scene->numSubdivMeshKeyFrames)
    {
      updateKeyFrame(g_ispc_scene);
      rtcCommit(g_scene);
      g_changed = true;
    }

#if  FIX_SAMPLING == 0
  g_accu_count++;
#endif

  if (camera_changed) {
    g_accu_count=0;
    memset(g_accu,0,width*height*sizeof(float4));

#if !defined(FORCE_FIXED_EDGE_TESSELLATION)
    if (g_subdiv_mode)
      {
       updateEdgeLevels(g_ispc_scene, cam_org);
       rtcCommit (g_scene);
      }
#endif

  }

  /* render image */
  const int numTilesX = (width +TILE_SIZE_X-1)/TILE_SIZE_X;
  const int numTilesY = (height+TILE_SIZE_Y-1)/TILE_SIZE_Y;
  launch[numTilesX*numTilesY] renderTile(pixels,width,height,time,vx,vy,vz,p,numTilesX,numTilesY); sync;
  //rtcDebug();
  COUNT_RAYS(PRINT(numRays));
} // device_render

/* called by the C++ code for cleanup */
export void device_cleanup ()
{
  delete[] g_accu;
  rtcDeleteScene (g_scene);
  rtcExit();
} // device_cleanup

#endif
