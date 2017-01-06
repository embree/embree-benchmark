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

#include "catmullclark_patch.h"
#include "bilinear_patch.h"
#include "bspline_patch.h"
#include "bezier_patch.h"
#include "gregory_patch.h"
#include "gregory_triangle_patch.h"
#include "tessellation_cache.h"

#if 1
#define PATCH_DEBUG_SUBDIVISION(x,y,z)
#else
#define PATCH_DEBUG_SUBDIVISION(x,y,z)                  \
  {                                                   \
    size_t hex = (size_t)this;                          \
    for (size_t i=0; i<4; i++) hex = hex ^ (hex >> 8);  \
    const float c = (float)(hex&0xff)/255.0f;           \
    if (P) *P = Vertex(0.5f+0.5f*x,0.5f+0.5f*y,0.5f+0.5f*z,0.0f);         \
    }               
#endif

#define PATCH_MAX_CACHE_DEPTH 4
#define PATCH_MAX_EVAL_DEPTH 8     // has to be larger or equal than PATCH_MAX_CACHE_DEPTH
#define PATCH_USE_GREGORY 1        // 0 = no gregory, 1 = fill, 2 = as early as possible
#define PATCH_USE_BEZIER_PATCH 0   // enable use of bezier instead of gregory patches

#if PATCH_USE_BEZIER_PATCH
#  define RegularPatch  BezierPatch
#  define RegularPatchT BezierPatchT<Vertex,Vertex_t>
#else
#  define RegularPatch  BSplinePatch
#  define RegularPatchT BSplinePatchT<Vertex,Vertex_t>
#endif

namespace embree
{
  template<typename vfloat>
  __forceinline Vec2<vfloat> map_tri_to_quad(const Vec2<vfloat>& uv)
  {
    const Vec2<vfloat> a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2<vfloat> ab = vfloat(0.5f)*(a+b), ac = vfloat(0.5f)*(a+c), bc = vfloat(0.5f)*(b+c), abc = vfloat(1.0f/3.0f)*(a+b+c);
    const Vec2<vfloat> A = a, B = ab-a, C = ac-a, D = a-ab-ac+abc;
    const vfloat AA = det(D,C), BB = det(D,A) + det(B,C) + det(uv,D), CC = det(B,A) + det(uv,B);
    const vfloat vv = (-BB+sqrt(BB*BB-4.0f*AA*CC))/(2.0f*AA);
    const vfloat uu = (uv.x - A.x - vv*C.x)/(B.x + vv*D.x);
    return Vec2<vfloat>(uu,vv);
  }
  
  template<typename vfloat>
    __forceinline Vec2<vfloat> map_quad_to_tri_dx(const Vec2f& a, const Vec2f& ab, const Vec2f& abc, const Vec2f& ac, const Vec2<vfloat>& xy) {
    return (1.0f-xy.y)*Vec2<vfloat>(ab-a) + xy.y*Vec2<vfloat>(abc-ac);
  }
  
  template<typename vfloat>
    __forceinline Vec2<vfloat> map_quad_to_tri_dy(const Vec2f& a, const Vec2f& ab, const Vec2f& abc, const Vec2f& ac, const Vec2<vfloat>& xy) {
    return (1.0f-xy.x)*Vec2<vfloat>(ac-a) + xy.x*Vec2<vfloat>(abc-ab);
  }
  
  template<typename vfloat>
    __forceinline auto right_of_line(const Vec2f& A, const Vec2f& B, const Vec2<vfloat>& P) -> decltype(P.x<P.x) {
    return det(Vec2<vfloat>(A)-P,Vec2<vfloat>(B)-P) <= 0.0f;
  }
  
  template<typename vfloat>
    __forceinline auto right_of_line_ab_abc(const Vec2<vfloat>& uv) -> decltype(uv.x<uv.x)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    return right_of_line(ab,abc,uv);
  }
  
  template<typename vfloat>
    __forceinline auto right_of_line_ac_abc(const Vec2<vfloat>& uv) -> decltype(uv.x<uv.x)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    return right_of_line(ac,abc,uv);
  }
  
  template<typename vfloat>
    __forceinline auto right_of_line_bc_abc(const Vec2<vfloat>& uv) -> decltype(uv.x<uv.x)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    return right_of_line(bc,abc,uv);
  }
  
  template<typename vfloat, typename Vertex>
    __forceinline void map_quad0_to_tri(const Vec2<vfloat>& xy, Vertex& dPdu, Vertex& dPdv)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    const Vertex dpdx = dPdu, dpdy = dPdv;
    const Vec2<vfloat> duvdx = map_quad_to_tri_dx(a,ab,abc,ac,xy);
    const Vec2<vfloat> duvdy = map_quad_to_tri_dy(a,ab,abc,ac,xy);
    const LinearSpace2<Vec2<vfloat> > J = rcp(LinearSpace2<Vec2<vfloat> >(duvdx,duvdy));
    dPdu = dpdx*J.vx.x + dpdy*J.vx.y;
    dPdv = dpdx*J.vy.x + dpdy*J.vy.y;
  }
  
  template<typename vfloat, typename Vertex>
    __forceinline void map_quad1_to_tri(const Vec2<vfloat>& xy, Vertex& dPdu, Vertex& dPdv)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    const Vertex dpdx = dPdu, dpdy = dPdv;
    const Vec2<vfloat> duvdx = map_quad_to_tri_dx(b,bc,abc,ab,xy);
    const Vec2<vfloat> duvdy = map_quad_to_tri_dy(b,bc,abc,ab,xy);
    const LinearSpace2<Vec2<vfloat> > J = rcp(LinearSpace2<Vec2<vfloat> >(duvdx,duvdy));
    dPdu = dpdx*J.vx.x + dpdy*J.vx.y;
    dPdv = dpdx*J.vy.x + dpdy*J.vy.y;
  }
  
  template<typename vfloat, typename Vertex>
    __forceinline void map_quad2_to_tri(const Vec2<vfloat>& xy, Vertex& dPdu, Vertex& dPdv)
  {
    const Vec2f a(0.0f,0.0f), b(1.0f,0.0f), c(0.0f,1.0f);
    const Vec2f ab = 0.5f*(a+b), ac = 0.5f*(a+c), bc = 0.5f*(b+c), abc = (1.0f/3.0f)*(a+b+c);
    const Vertex dpdx = dPdu, dpdy = dPdv;
    const Vec2<vfloat> duvdx = map_quad_to_tri_dx(c,ac,abc,bc,xy);
    const Vec2<vfloat> duvdy = map_quad_to_tri_dy(c,ac,abc,bc,xy);
    const LinearSpace2<Vec2<vfloat> > J = rcp(LinearSpace2<Vec2<vfloat> >(duvdx,duvdy));
    dPdu = dpdx*J.vx.x + dpdy*J.vx.y;
    dPdv = dpdx*J.vy.x + dpdy*J.vy.y;
  }
  
  template<typename vbool, typename vfloat>
    __forceinline void map_quad0_to_tri(const vbool& valid, const Vec2<vfloat>& xy, float* dPdu, float* dPdv, size_t dstride, size_t i)
  {
    vfloat dPdut = vfloat::loadu(dPdu+i*dstride), dPdvt = vfloat::loadu(dPdv+i*dstride);
    map_quad0_to_tri(xy,dPdut,dPdvt); 
    vfloat::store(valid,dPdu+i*dstride,dPdut);
    vfloat::store(valid,dPdv+i*dstride,dPdvt);
  }

  template<typename vbool, typename vfloat>
    __forceinline void map_quad1_to_tri(const vbool& valid, const Vec2<vfloat>& xy, float* dPdu, float* dPdv, size_t dstride, size_t i)
  {
    vfloat dPdut = vfloat::loadu(dPdu+i*dstride), dPdvt = vfloat::loadu(dPdv+i*dstride);
    map_quad1_to_tri(xy,dPdut,dPdvt); 
    vfloat::store(valid,dPdu+i*dstride,dPdut);
    vfloat::store(valid,dPdv+i*dstride,dPdvt);
  }

  template<typename vbool, typename vfloat>
    __forceinline void map_quad2_to_tri(const vbool& valid, const Vec2<vfloat>& xy, float* dPdu, float* dPdv, size_t dstride, size_t i)
  {
    vfloat dPdut = vfloat::loadu(dPdu+i*dstride), dPdvt = vfloat::loadu(dPdv+i*dstride);
    map_quad2_to_tri(xy,dPdut,dPdvt); 
    vfloat::store(valid,dPdu+i*dstride,dPdut);
    vfloat::store(valid,dPdv+i*dstride,dPdvt);
  }

  template<typename Vertex, typename Vertex_t = Vertex>
    struct __aligned(64) PatchT
    {
    public:
    
    typedef GeneralCatmullClarkPatchT<Vertex,Vertex_t> GeneralCatmullClarkPatch;
    typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
    
    enum Type {
      INVALID_PATCH = 0,
      BILINEAR_PATCH = 1,
      BSPLINE_PATCH = 2,  
      BEZIER_PATCH = 3,  
      GREGORY_PATCH = 4,
      SUBDIVIDED_GENERAL_TRIANGLE_PATCH = 5,
      SUBDIVIDED_GENERAL_QUAD_PATCH = 6,
      SUBDIVIDED_GENERAL_PATCH = 7,
      SUBDIVIDED_QUAD_PATCH = 8
    };

    struct BilinearPatch 
    {
      /* creates BilinearPatch from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static BilinearPatch* create(const Allocator& alloc, const CatmullClarkPatch& patch) {
        return new (alloc(sizeof(BilinearPatch))) BilinearPatch(patch);
      }
      
      __forceinline BilinearPatch (const CatmullClarkPatch& patch) 
        : type(BILINEAR_PATCH), patch(patch) {}
      
    public:
      Type type;
      BilinearPatchT<Vertex,Vertex_t> patch;
    };
    
    struct BSplinePatch 
    {
      /* creates BSplinePatch from a half edge */
      template<typename Loader, typename Allocator>
        __noinline static BSplinePatch* create(const Allocator& alloc, const SubdivMesh::HalfEdge* edge, const Loader& loader) {
        return new (alloc(sizeof(BSplinePatch))) BSplinePatch(edge,loader);
      }
      
      template<typename Loader>
      __forceinline BSplinePatch (const SubdivMesh::HalfEdge* edge, const Loader& loader) 
      : type(BSPLINE_PATCH), patch(edge,loader) {}
      
      /* creates BSplinePatch from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static BSplinePatch* create(const Allocator& alloc, const CatmullClarkPatch& patch) {
        return new (alloc(sizeof(BSplinePatch))) BSplinePatch(patch);
      }
      
      __forceinline BSplinePatch (const CatmullClarkPatch& patch) 
        : type(BSPLINE_PATCH), patch(patch) {}
      
    public:
      Type type;
      BSplinePatchT<Vertex,Vertex_t> patch;
    };

    struct BezierPatch
    {
      /* creates BezierPatch from a half edge */
      template<typename Loader, typename Allocator>
        __noinline static BezierPatch* create(const Allocator& alloc, const SubdivMesh::HalfEdge* edge, const Loader& loader) {
        return new (alloc(sizeof(BezierPatch))) BezierPatch(edge,loader);
      }
      
      template<typename Loader>
      __forceinline BezierPatch (const SubdivMesh::HalfEdge* edge, const Loader& loader) 
      : type(BEZIER_PATCH), patch(edge,loader) {}
      
      /* creates Bezier from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static BezierPatch* create(const Allocator& alloc, const CatmullClarkPatch& patch) {
        return new (alloc(sizeof(BezierPatch))) BezierPatch(patch);
      }
      
      __forceinline BezierPatch (const CatmullClarkPatch& patch) 
        : type(BEZIER_PATCH), patch(patch) {}
      
    public:
      Type type;
      BezierPatchT<Vertex,Vertex_t> patch;
    };
    
    struct GregoryPatch
    {
      /* creates GregoryPatch from half edge */
      template<typename Loader, typename Allocator>
        __noinline static GregoryPatch* create(const Allocator& alloc, const SubdivMesh::HalfEdge* edge, const Loader& loader) {
        return new (alloc(sizeof(GregoryPatch))) GregoryPatch(edge,loader);
      }
      
      template<typename Loader>
      __forceinline GregoryPatch (const SubdivMesh::HalfEdge* edge, const Loader& loader) 
      : type(GREGORY_PATCH) { CatmullClarkPatch ccpatch; ccpatch.init2(edge,loader); patch.init(ccpatch); }
      
      /* creates GregoryPatch from CatmullClarkPatch */
      template<typename Allocator>
      __noinline static GregoryPatch* create(const Allocator& alloc, const CatmullClarkPatch& patch) {
        return new (alloc(sizeof(GregoryPatch))) GregoryPatch(patch);
      }
      
      __forceinline GregoryPatch (const CatmullClarkPatch& patch) 
        : type(GREGORY_PATCH), patch(patch) {}
      
    public:
      Type type;
      GregoryPatchT<Vertex,Vertex_t> patch;
    };
    
    struct SubdividedGeneralTrianglePatch
    {
      template<typename Allocator>
      __noinline static SubdividedGeneralTrianglePatch* create(const Allocator& alloc, PatchT* children[3]) {
        return new (alloc(sizeof(SubdividedGeneralTrianglePatch))) SubdividedGeneralTrianglePatch(children);
      }
      
      __forceinline SubdividedGeneralTrianglePatch(PatchT* children[3]) : type(SUBDIVIDED_GENERAL_TRIANGLE_PATCH) {
        for (size_t i=0; i<3; i++) child[i] = children[i];
      }
      
      Type type;
      PatchT* child[3];
    };
    
    struct SubdividedQuadPatch
    {
      template<typename Allocator>
      __noinline static SubdividedQuadPatch* create(const Allocator& alloc, PatchT* children[4]) {
        return new (alloc(sizeof(SubdividedQuadPatch))) SubdividedQuadPatch(children);
      }
      
      __forceinline SubdividedQuadPatch(PatchT* children[4]) : type(SUBDIVIDED_QUAD_PATCH) {
        for (size_t i=0; i<4; i++) child[i] = children[i];
      }
      
    public:
      Type type;
      PatchT* child[4];
    };
    
    struct SubdividedGeneralQuadPatch
    {
      template<typename Allocator>
      __noinline static SubdividedGeneralQuadPatch* create(const Allocator& alloc, PatchT* children[4]) {
        return new (alloc(sizeof(SubdividedGeneralQuadPatch))) SubdividedGeneralQuadPatch(children);
      }
      
      __forceinline SubdividedGeneralQuadPatch(PatchT* children[4]) : type(SUBDIVIDED_GENERAL_QUAD_PATCH) {
        for (size_t i=0; i<4; i++) child[i] = children[i];
      }
      
      Type type;
      PatchT* child[4];
    };

    struct SubdividedGeneralPatch
    {
      template<typename Allocator>
      __noinline static SubdividedGeneralPatch* create(const Allocator& alloc, PatchT** children, const size_t N) {
        return new (alloc(sizeof(SubdividedGeneralPatch))) SubdividedGeneralPatch(children,N);
      }
      
      __forceinline SubdividedGeneralPatch(PatchT** children, const size_t N) : type(SUBDIVIDED_GENERAL_PATCH), N(N) {
        for (size_t i=0; i<N; i++) child[i] = children[i];
      }
      
      Type type;
      size_t N;
      PatchT* child[SubdivMesh::MAX_VALENCE];
    };
    
    /*! Default constructor. */
    __forceinline PatchT () {}
    
    template<typename Allocator>
    __noinline static PatchT* create(const Allocator& alloc, const SubdivMesh::HalfEdge* edge, const char* vertices, size_t stride)
    {
      auto loader = [&](const SubdivMesh::HalfEdge* p) -> Vertex { 
        const unsigned vtx = p->getStartVertexIndex();
        return Vertex_t::loadu((float*)&vertices[vtx*stride]);
      };
      
      if (PATCH_MAX_CACHE_DEPTH == 0) 
        return nullptr;

      PatchT* child = nullptr;
      switch (edge->patch_type) {
      case SubdivMesh::REGULAR_QUAD_PATCH:   child = (PatchT*) RegularPatch::create(alloc,edge,loader); break;
#if PATCH_USE_GREGORY == 2
      case SubdivMesh::IRREGULAR_QUAD_PATCH: child = (PatchT*) GregoryPatch::create(alloc,edge,loader); break;
#endif
      default: {
        GeneralCatmullClarkPatch patch(edge,loader);
        child = (PatchT*) PatchT::create(alloc,patch,edge,vertices,stride,0);
      }
      }
      return child;
    }

    template<typename Allocator>
    __noinline static PatchT* create(const Allocator& alloc, GeneralCatmullClarkPatch& patch, const SubdivMesh::HalfEdge* edge, const char* vertices, size_t stride, size_t depth)
    {
      /* convert into standard quad patch if possible */
      if (likely(patch.isQuadPatch())) 
      {
        CatmullClarkPatch qpatch; patch.init(qpatch);
        return PatchT::create(alloc,qpatch,edge,vertices,stride,depth);
      }
      
      if (depth >= PATCH_MAX_CACHE_DEPTH)
        return nullptr;
      
      /* subdivide patch */
      size_t N;
      array_t<CatmullClarkPatch,GeneralCatmullClarkPatch::SIZE> patches; 
      patch.subdivide(patches,N);
      
      if (N == 3) 
      {
        PatchT* child[3];
        for (size_t i=0; i<3; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);

        return (PatchT*) SubdividedGeneralTrianglePatch::create(alloc,child);
      } 
      else if (N == 4) 
      {
        PatchT* child[4];
        for (size_t i=0; i<4; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
        return (PatchT*) SubdividedGeneralQuadPatch::create(alloc,child);
      }
      else 
      {
        assert(N<SubdivMesh::MAX_VALENCE);
        PatchT* child[SubdivMesh::MAX_VALENCE];
        for (size_t i=0; i<N; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
        return (PatchT*) SubdividedGeneralPatch::create(alloc,child,N);
      }
      
      return nullptr;
    }

    template<typename Allocator>
    __noinline static PatchT* create(const Allocator& alloc, CatmullClarkPatch& patch, const SubdivMesh::HalfEdge* edge, const char* vertices, size_t stride, size_t depth)
    {
      if (unlikely(patch.isRegular2())) { 
        assert(depth > 0); return (PatchT*) RegularPatch::create(alloc,patch); 
      }
#if PATCH_USE_GREGORY == 2
      else if (unlikely(depth>=PATCH_MAX_EVAL_DEPTH || patch.isGregory())) { 
        assert(depth > 0); return (PatchT*) GregoryPatch::create(alloc,patch); 
      }
#else
      else if (unlikely(depth>=PATCH_MAX_EVAL_DEPTH))
      {
#if PATCH_USE_GREGORY == 1
        return (PatchT*) GregoryPatch::create(alloc,patch); 
#else
        return (PatchT*) BilinearPatch::create(alloc,patch);
#endif
      }
#endif
      else if (depth >= PATCH_MAX_CACHE_DEPTH) 
        return nullptr;
      
      else 
      {
        PatchT* child[4];
        array_t<CatmullClarkPatch,4> patches; 
        patch.subdivide(patches);
        for (size_t i=0; i<4; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
        return (PatchT*) SubdividedQuadPatch::create(alloc,child);
      }
    }

  public:
    Type type;
  };
}
