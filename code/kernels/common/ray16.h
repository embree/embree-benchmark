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

#include "ray.h"

namespace embree
{
  /*! Ray structure. Contains all information of 16 rays including
   *  precomputed reciprocal direction. */
  struct Ray16
  {
    typedef bool16 simdb;
    typedef float16 simdf;
    typedef int16 simdi;

    /*! Default construction does nothing. */
    __forceinline Ray16() {}

    /*! Constructs a ray from origin, direction, and ray segment. Near
     *  has to be smaller than far. */
    __forceinline Ray16(const Vec3f16& org, const Vec3f16& dir, 
                        const float16& tnear = zero, const float16& tfar = inf, 
                        const float16& time = zero, const int16& mask = -1)
      : org(org), dir(dir), tnear(tnear), tfar(tfar), geomID(-1), primID(-1), mask(mask), time(time) {}

    /*! Tests if we hit something. */
    __forceinline operator bool16() const { return geomID != int16(-1); }

  public:
    Vec3f16 org;      //!< Ray origin
    Vec3f16 dir;      //!< Ray direction
    float16 tnear;    //!< Start of ray segment 
    float16 tfar;     //!< End of ray segment   
    float16 time;     //!< Time of this ray for motion blur.
    int16 mask;     //!< used to mask out objects during traversal

    Vec3f16 Ng;       //!< Geometry normal
    float16 u;        //!< Barycentric u coordinate of hit
    float16 v;        //!< Barycentric v coordinate of hit
    int16 geomID;   //!< geometry ID
    int16 primID;   //!< primitive ID
    int16 instID;   //!< instance ID

    template<int PFHINT>
    __forceinline void prefetchHitData() const
    {
      prefetch<PFHINT>(&geomID);
      prefetch<PFHINT>(&primID);
      prefetch<PFHINT>(&tfar);
      prefetch<PFHINT>(&u);
      prefetch<PFHINT>(&v);
      prefetch<PFHINT>(&Ng.x);
      prefetch<PFHINT>(&Ng.y);
      prefetch<PFHINT>(&Ng.z);
    }

    __forceinline void update(const bool16 &m_mask,
			      const size_t rayIndex,
			      const float16 &new_t,
			      const float16 &new_u,
			      const float16 &new_v,
			      const float16 &new_gnormalx,
			      const float16 &new_gnormaly,
			      const float16 &new_gnormalz,
			      const int new_geomID,
			      const int new_primID)
    {
      geomID[rayIndex] = new_geomID;
      primID[rayIndex] = new_primID;

      compactustore16f_low(m_mask,&tfar[rayIndex],new_t);
      compactustore16f_low(m_mask,&u[rayIndex],new_u); 
      compactustore16f_low(m_mask,&v[rayIndex],new_v); 
      compactustore16f_low(m_mask,&Ng.x[rayIndex],new_gnormalx); 
      compactustore16f_low(m_mask,&Ng.y[rayIndex],new_gnormaly); 
      compactustore16f_low(m_mask,&Ng.z[rayIndex],new_gnormalz);       

    }

    __forceinline void update(const bool16 &m_mask,
			      const float16 &new_t,
			      const float16 &new_u,
			      const float16 &new_v,
			      const float16 &new_gnormalx,
			      const float16 &new_gnormaly,
			      const float16 &new_gnormalz,
			      const int16 &new_geomID,
			      const int16 &new_primID)
    {
      store16f(m_mask,(float*)&tfar,new_t);
      store16f(m_mask,(float*)&u,new_u);
      store16f(m_mask,(float*)&v,new_v);
      store16f(m_mask,(float*)&Ng.x,new_gnormalx);
      store16f(m_mask,(float*)&Ng.y,new_gnormaly);
      store16f(m_mask,(float*)&Ng.z,new_gnormalz);
      store16i(m_mask,(int*)&geomID,new_geomID);
      store16i(m_mask,(int*)&primID,new_primID);     
    }
  };

  /*! Outputs ray to stream. */
  inline std::ostream& operator<<(std::ostream& cout, const Ray16& ray) {
    return cout << "{ " << std::endl 
                << "org = " << ray.org << std::endl 
                << " dir = " << ray.dir << std::endl 
                << " near = " << ray.tnear << std::endl 
                << " far = " << ray.tfar << std::endl 
                << " time = " << ray.time << std::endl 
                << " instID = " << ray.instID << std::endl 
                << " geomID = " << ray.geomID << std::endl 
                << " primID = " << ray.primID <<  std::endl
                << " u = " << ray.u <<  std::endl 
                << " v = " << ray.v << std::endl 
                << " Ng = " << ray.Ng 
                << " }";
  }
}
