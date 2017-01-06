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

#include "../scene_subdiv_mesh.h"
#include "subdivpatch1base.h"
#include "../scene.h"

namespace embree
{

  __forceinline Vec3fa computeInnerBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x)
  {
    return 1.0f / 36.0f * (16.0f * v[y][x] + 4.0f * (v[y-1][x] +  v[y+1][x] + v[y][x-1] + v[y][x+1]) + (v[y-1][x-1] + v[y+1][x+1] + v[y-1][x+1] + v[y+1][x-1]));
  }
  
  __forceinline Vec3fa computeTopEdgeBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x)
  {
    return 1.0f / 18.0f * (8.0f * v[y][x] + 4.0f * v[y-1][x] + 2.0f * (v[y][x-1] + v[y][x+1]) + (v[y-1][x-1] + v[y-1][x+1]));
  }

  __forceinline Vec3fa computeButtomEdgeBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x)
  {
    return 1.0f / 18.0f * (8.0f * v[y][x] + 4.0f * v[y+1][x] + 2.0f * (v[y][x-1] + v[y][x+1]) + v[y+1][x-1] + v[y+1][x+1]);
  }

  __forceinline Vec3fa computeLeftEdgeBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x)
  {
    return 1.0f / 18.0f * (8.0f * v[y][x] + 4.0f * v[y][x-1] + 2.0f * (v[y-1][x] + v[y+1][x]) + v[y-1][x-1] + v[y+1][x-1]);
  }

  __forceinline Vec3fa computeRightEdgeBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x)
  {
    return 1.0f / 18.0f * (8.0f * v[y][x] + 4.0f * v[y][x+1] + 2.0f * (v[y-1][x] + v[y+1][x]) + v[y-1][x+1] + v[y+1][x+1]);
  }

  __forceinline Vec3fa computeCornerBezierControlPoint(const Vec3fa v[4][4], const size_t y, const size_t x, const ssize_t delta_y, const ssize_t delta_x)
  {
    return 1.0f / 9.0f * (4.0f * v[y][x] + 2.0f * (v[y+delta_y][x] + v[y][x+delta_x]) + v[y+delta_y][x+delta_x]);
  }

  void convertToBicubicBezierPatch(const Vec3fa source[4][4],
                                   Vec3fa dest[4][4])
  {
    /* compute inner bezier control points */
    dest[0][0] = computeInnerBezierControlPoint(source,1,1);
    dest[0][3] = computeInnerBezierControlPoint(source,1,2);
    dest[3][3] = computeInnerBezierControlPoint(source,2,2);
    dest[3][0] = computeInnerBezierControlPoint(source,2,1);

    /* compute top edge control points */
    dest[0][1] = computeRightEdgeBezierControlPoint(source,1,1);
    dest[0][2] = computeLeftEdgeBezierControlPoint(source,1,2); 
 
    /* compute buttom edge control points */
    dest[3][1] = computeRightEdgeBezierControlPoint(source,2,1);
    dest[3][2] = computeLeftEdgeBezierControlPoint(source,2,2);
    
    /* compute left edge control points */
    dest[1][0] = computeButtomEdgeBezierControlPoint(source,1,1);
    dest[2][0] = computeTopEdgeBezierControlPoint(source,2,1);

    /* compute right edge control points */
    dest[1][3] = computeButtomEdgeBezierControlPoint(source,1,2);
    dest[2][3] = computeTopEdgeBezierControlPoint(source,2,2);

    /* compute corner control points */
    dest[1][1] = computeCornerBezierControlPoint(source,1,1, 1, 1);
    dest[1][2] = computeCornerBezierControlPoint(source,1,2, 1,-1);
    dest[2][2] = computeCornerBezierControlPoint(source,2,2,-1,-1);
    dest[2][1] = computeCornerBezierControlPoint(source,2,1,-1, 1);        
  }
  
  /*! Construction from vertices and IDs. */
  SubdivPatch1Base::SubdivPatch1Base (const CatmullClarkPatch3fa& ipatch,
                                      const int fas_depth,
                                      const unsigned int gID,
                                      const unsigned int pID,
                                      const SubdivMesh *const mesh,
                                      const Vec2f uv[4],
                                      const float edge_level[4],
                                      const int neighborSubdiv[4],
                                      const BezierCurve3fa *border, 
                                      const int border_flags) 
    : geom(gID),prim(pID),flags(0)
  {
    static_assert(sizeof(SubdivPatch1Base) == 5 * 64, "SubdivPatch1Base has wrong size");
    mtx.reset();

    for (size_t i=0; i<4; i++) {
      u[i] = (unsigned short)(uv[i].x * 65535.0f);
      v[i] = (unsigned short)(uv[i].y * 65535.0f);
    }

    updateEdgeLevels(edge_level,mesh);
    
    /* determine whether patch is regular or not */
    if (fas_depth == 0 && ipatch.isRegular1() && !ipatch.hasBorder()) /* only select b-spline/bezier in the interior and not FAS-based patches*/
    {
#if 0
      /* bezier */
      BSplinePatch3fa tmp;
      tmp.init( ipatch );
      convertToBicubicBezierPatch(tmp.v,patch.v);
      flags |= BEZIER_PATCH;
#else
      /* bspline */

      flags |= BSPLINE_PATCH;
      patch.init( ipatch );
#endif      
    }
    else
    {
      /* gregory patches */
      flags |= GREGORY_PATCH;
      GregoryPatch3fa gpatch; 
      gpatch.init_crackfix( ipatch, fas_depth, neighborSubdiv, border, border_flags ); 
      gpatch.exportDenseConrolPoints( patch.v );
    }
  }
  

  void SubdivPatch1Base::updateEdgeLevels(const float edge_level[4],const SubdivMesh *const mesh)
  {
    /* init discrete edge tessellation levels and grid resolution */

    assert( edge_level[0] >= 0.0f );
    assert( edge_level[1] >= 0.0f );
    assert( edge_level[2] >= 0.0f );
    assert( edge_level[3] >= 0.0f );

#if defined(__MIC__)
    const size_t SIMD_WIDTH = 16;
#else
    const size_t SIMD_WIDTH = 8;
#endif

    level[0] = max(ceilf(edge_level[0]),1.0f);
    level[1] = max(ceilf(edge_level[1]),1.0f);
    level[2] = max(ceilf(edge_level[2]),1.0f);
    level[3] = max(ceilf(edge_level[3]),1.0f);

    grid_u_res = max(level[0],level[2])+1; // n segments -> n+1 points
    grid_v_res = max(level[1],level[3])+1;
    
    /* workaround for 2x2 intersection stencil */
#if !defined(__MIC__)    
    grid_u_res = max(grid_u_res,3);
    grid_v_res = max(grid_v_res,3);
#endif

#if defined(__MIC__)    
    grid_size_simd_blocks        = ((grid_u_res*grid_v_res+15)&(-16)) / 16;
    grid_subtree_size_64b_blocks = 5; // single leaf with u,v,x,y,z      
#else
    /* 8-wide SIMD is default on Xeon */
    grid_size_simd_blocks        = ((grid_u_res*grid_v_res+7)&(-8)) / 8;
    grid_subtree_size_64b_blocks = (sizeof(Quad2x2)+63) / 64; // single Quad2x2

#endif
    /* need stiching? */

    flags &= ~TRANSITION_PATCH;

    const unsigned int int_edge_points0 = (unsigned int)level[0] + 1;
    const unsigned int int_edge_points1 = (unsigned int)level[1] + 1;
    const unsigned int int_edge_points2 = (unsigned int)level[2] + 1;
    const unsigned int int_edge_points3 = (unsigned int)level[3] + 1;
      
    if (int_edge_points0 < (unsigned int)grid_u_res ||
	int_edge_points2 < (unsigned int)grid_u_res ||
	int_edge_points1 < (unsigned int)grid_v_res ||
	int_edge_points3 < (unsigned int)grid_v_res)
      flags |= TRANSITION_PATCH;

    /* tessellate into grid blocks for larger grid resolutions, generate bvh4 subtree over grid blocks*/

#if defined(__MIC__)
    const size_t leafBlocks = 4;
#else
    const size_t leafBlocks = (sizeof(Quad2x2)+63) / 64;
#endif
    grid_bvh_size_64b_blocks = getSubTreeSize64bBlocks( 0 );
    
    const size_t grid_size_xyzuv = (grid_size_simd_blocks * SIMD_WIDTH) * 4;
    grid_subtree_size_64b_blocks = grid_bvh_size_64b_blocks + ((grid_size_xyzuv+15) / 16);


    /* has displacements? */
    flags &= ~HAS_DISPLACEMENT;
    if (mesh->displFunc != nullptr)
      flags |= HAS_DISPLACEMENT;

  }



  void SubdivPatch1Base::evalToOBJ(Scene *scene,size_t &vertex_index, size_t &numTotalTriangles)
  {
#if defined(__MIC__)
    THROW_RUNTIME_ERROR("EVALTOOBJ NOT SUPPORTED ON MIC");
#else

#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __aligned(64) float grid_x[(grid_size_simd_blocks+1)*8]; 
    __aligned(64) float grid_y[(grid_size_simd_blocks+1)*8];
    __aligned(64) float grid_z[(grid_size_simd_blocks+1)*8]; 
    
    __aligned(64) float grid_u[(grid_size_simd_blocks+1)*8]; 
    __aligned(64) float grid_v[(grid_size_simd_blocks+1)*8];

    __aligned(64) float grid_nx[(grid_size_simd_blocks+1)*8]; 
    __aligned(64) float grid_ny[(grid_size_simd_blocks+1)*8];
    __aligned(64) float grid_nz[(grid_size_simd_blocks+1)*8]; 
 
#else
      const size_t array_elements = (grid_size_simd_blocks + 1) * 8;
      float *const ptr = (float*)_malloca(8 * array_elements * sizeof(float) + 64);
      float *const grid_arrays = (float*)ALIGN_PTR(ptr,64);

      float *grid_x = &grid_arrays[array_elements * 0];
      float *grid_y = &grid_arrays[array_elements * 1];
      float *grid_z = &grid_arrays[array_elements * 2];
      float *grid_u = &grid_arrays[array_elements * 3];
      float *grid_v = &grid_arrays[array_elements * 4];

      float *grid_nx = &grid_arrays[array_elements * 5];
      float *grid_ny = &grid_arrays[array_elements * 6];
      float *grid_nz = &grid_arrays[array_elements * 7];

#endif
      SubdivMesh *mesh = (SubdivMesh *)scene->getSubdivMesh(geom);
  
      evalGrid(*this,grid_x,grid_y,grid_z,grid_u,grid_v,mesh);

#if defined(__AVX__)
    for (size_t i=0;i<grid_size_simd_blocks;i++)
      {
        float8 uu = load8f(&grid_u[8*i]);
        float8 vv = load8f(&grid_v[8*i]);
        Vec3f8 normal = normalize(patch.normal(uu,vv));
        *(float8*)&grid_nx[8*i] = normal.x;
        *(float8*)&grid_ny[8*i] = normal.y;
        *(float8*)&grid_nz[8*i] = normal.z;        
      }
#else
    for (size_t i=0;i<grid_size_simd_blocks*2;i++) // 4-wide blocks for SSE
      {
        float4 uu      = load4f(&grid_u[4*i]);
        float4 vv      = load4f(&grid_v[4*i]);
        Vec3f4 normal = normalize(patch.normal(uu,vv));
        *(float4*)&grid_nx[4*i] = normal.x;
        *(float4*)&grid_ny[4*i] = normal.y;
        *(float4*)&grid_nz[4*i] = normal.z;        
      }
#endif
      
      std::cout << "# grid_u_res " << grid_u_res << std::endl;
      std::cout << "# grid_v_res " << grid_v_res << std::endl;

      /* vertex */
      for (size_t v=0;v<grid_v_res;v++)
        for (size_t u=0;u<grid_u_res;u++)
          {
            size_t offset = v * grid_u_res + u;
            std::cout << "v " << grid_x[offset] << " " << grid_y[offset] << " " << grid_z[offset] << std::endl;
          }

      /* normal */
      for (size_t v=0;v<grid_v_res;v++)
        for (size_t u=0;u<grid_u_res;u++)
          {
            size_t offset = v * grid_u_res + u;
            std::cout << "vn " << grid_nx[offset] << " " << grid_ny[offset] << " " << grid_nz[offset] << std::endl;
          }

      for (size_t v=0;v<grid_v_res-1;v++)
        for (size_t u=0;u<grid_u_res-1;u++)
          {
            size_t offset0 = vertex_index + v * grid_u_res + u;
            size_t offset1 = offset0 + grid_u_res;
            //std::cout << "f " << offset0+1 << " " << offset0+2 << " " << offset1+2 << " " << offset1+1 << std::endl;
            std::cout << "f " << offset0+1 << "//" << offset0+1 << " " << offset0+2<< "//" << offset0+2 << " " << offset1+2<< "//" << offset1+2 << " " << offset1+1<< "//" << offset1+1 << std::endl;
	    numTotalTriangles += 2;
          }
      vertex_index += grid_u_res*grid_v_res;
      
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
      _freea(ptr);
#endif      

#endif
  }

}
