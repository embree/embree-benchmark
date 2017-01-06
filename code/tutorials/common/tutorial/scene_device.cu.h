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

struct ISPCTriangle 
{
  int v0;                /*< first triangle vertex */
  int v1;                /*< second triangle vertex */
  int v2;                /*< third triangle vertex */
  int materialID;        /*< material of triangle */
};

struct ISPCQuad
{
  int v0;                /*< first triangle vertex */
  int v1;                /*< second triangle vertex */
  int v2;                /*< third triangle vertex */
  int v3;                /*< fourth triangle vertex */
};

struct ISPCMesh
{
  float4* positions;    //!< vertex position array
  float4* positions2;    //!< vertex position array
  float4* normals;       //!< vertex normal array
  float2* texcoords;     //!< vertex texcoord array
  ISPCTriangle* triangles;  //!< list of triangles
  ISPCQuad* quads;  //!< list of triangles
  float *edge_level; // FIXME: remove this
  int numVertices;
  int numTriangles;
  int numQuads;
  int geomID;
  int meshMaterialID;
};

struct ISPCSubdivMesh
{
  float4* positions;       //!< vertex positions
  float4* normals;         //!< face vertex normals
  float2* texcoords;        //!< face texture coordinates
  int* position_indices;   //!< position indices for all faces
  int* normal_indices;     //!< normal indices for all faces
  int* texcoord_indices;   //!< texcoord indices for all faces
  int* verticesPerFace;    //!< number of indices of each face
  int* holes;              //!< face ID of holes
  float* subdivlevel;      //!< subdivision level
  Vec2i* edge_creases;          //!< crease index pairs
  float* edge_crease_weights;   //!< weight for each crease
  int* vertex_creases;          //!< indices of vertex creases
  float* vertex_crease_weights; //!< weight for each vertex crease
  int* face_offsets;
  int numVertices;
  int numFaces;
  int numEdges;
  int numEdgeCreases;
  int numVertexCreases;
  int numHoles;
  int materialID;
  int geomID;
};

struct ISPCHair
{
  int vertex;
  int id;
};

struct ISPCHairSet
{
  float4* v;       //!< hair control points (x,y,z,r)
  float4* v2;       //!< hair control points (x,y,z,r)
  ISPCHair* hairs; //!< for each hair, index to first control point
  int numVertices;
  int numHairs;
};

struct ISPCAmbientLight
{
  float4 L;                  //!< radiance of ambient light
};

struct ISPCPointLight
{
  float4 P;                  //!< position of point light
  float4 I;                  //!< radiant intensity of point light
};

struct ISPCDirectionalLight
{
  float4 D;                  //!< Light direction
  float4 E;                  //!< Irradiance (W/m^2)
};

struct ISPCDistantLight
{
  float4 D;             //!< Light direction
  float4 L;             //!< Radiance (W/(m^2*sr))
  float halfAngle;     //!< Half illumination angle
  float radHalfAngle;  //!< Half illumination angle in radians
  float cosHalfAngle;  //!< Cosine of half illumination angle
};

enum MaterialTy { MATERIAL_OBJ, MATERIAL_THIN_DIELECTRIC, MATERIAL_METAL, MATERIAL_VELVET, MATERIAL_DIELECTRIC, MATERIAL_METALLIC_PAINT, MATERIAL_MATTE, MATERIAL_MIRROR, MATERIAL_REFLECTIVE_METAL };

struct ISPCMaterial
{
  int ty;
  int align0,align1,align2;
  float4 v[7];
};

struct MatteMaterial
{
  int ty;
  int align[3];

  float4 reflectance;
};

struct MirrorMaterial
{
  int ty;
  int align[3];
  float4 reflectance;
};

enum TEXTURE_FORMAT {
  RGBA8        = 1,
  RGB8         = 2,
  FLOAT32      = 3,
  PTEX_RGBA8   = 4,
  PTEX_FLOAT32 = 5
};

struct Texture {      
  int width;
  int height;    
  int format;
  int bytesPerTexel;
  int width_mask;
  int height_mask;
  void *data;
};

struct OBJMaterial
{
  int ty;
  int align[3];

  int illum;             /*< illumination model */
  float d;               /*< dissolve factor, 1=opaque, 0=transparent */
  float Ns;              /*< specular exponent */
  float Ni;              /*< optical density for the surface (index of refraction) */
  
  float4 Ka;              /*< ambient reflectivity */
  float4 Kd;              /*< diffuse reflectivity */
  float4 Ks;              /*< specular reflectivity */
  float4 Kt;              /*< transmission filter */

  Texture* map_Kd;       /*< dummy */
  Texture* map_Displ;       /*< dummy */
};

struct MetalMaterial
{
  int ty;
  int align[3];

  float4 reflectance;
  float4 eta;
  float4 k;
  float roughness;
};

struct ReflectiveMetalMaterial
{
  int ty;
  int align[3];

  float4 reflectance;
  float4 eta;
  float4 k;
  float roughness;
};

struct VelvetMaterial
{
  int ty;
  int align[3];

  float4 reflectance;
  float4 horizonScatteringColor;
  float backScattering;
  float horizonScatteringFallOff;
};

struct DielectricMaterial
{
  int ty;
  int align[3];
  float4 transmissionOutside;
  float4 transmissionInside;
  float etaOutside;
  float etaInside;
};

struct ThinDielectricMaterial
{
  int ty;
  int align[3];
  float4 transmission;
  float eta;
};

struct MetallicPaintMaterial
{
  int ty;
  int align[3];
  float4 shadeColor;
  float4 glitterColor;
  float glitterSpread;
  float eta;
};

struct ISPCSubdivMeshKeyFrame {
  ISPCSubdivMesh** subdiv;                   //!< list of subdiv meshes
  int numSubdivMeshes;                       //!< number of subdiv meshes
};



  //ISPCMesh** meshes;   //!< list of meshes
  rtBuffer<ISPCMaterial> materials; //     //!< material list
  int numMeshes;                       //!< number of meshes
  int numMaterials;                    //!< number of materials

  ISPCHairSet** hairs;
  int numHairSets;

  rtBuffer<ISPCAmbientLight> ambientLights; //!< list of ambient lights
  rtDeclareVariable(int, numAmbientLights, , );                    //!< number of ambient lights
  
  rtBuffer<ISPCPointLight> pointLights;     //!< list of point lights
  rtDeclareVariable(int, numPointLights, , );                      //!< number of point lights
  
  rtBuffer<ISPCDirectionalLight> dirLights; //!< list of directional lights
  rtDeclareVariable(int, numDirectionalLights, , );                //!< number of directional lights

  rtBuffer<ISPCDistantLight> distantLights; //!< list of distant lights
  rtDeclareVariable(int, numDistantLights, , );                    //!< number of distant lights

  ISPCSubdivMesh** subdiv;                   //!< list of subdiv meshes
  int numSubdivMeshes;                       //!< number of subdiv meshes

  ISPCSubdivMeshKeyFrame** subdivMeshKeyFrames;
  int numSubdivMeshKeyFrames;

