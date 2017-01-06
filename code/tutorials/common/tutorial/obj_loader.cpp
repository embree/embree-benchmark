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

#include "obj_loader.h"
#include "texture_loader.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <string.h>

#define FORCE_ONLY_QUADS 0

namespace embree
{
  /*! Three-index vertex, indexing start at 0, -1 means invalid vertex. */
  struct Vertex {
    int v, vt, vn;
    Vertex() {};
    Vertex(int v) : v(v), vt(v), vn(v) {};
    Vertex(int v, int vt, int vn) : v(v), vt(vt), vn(vn) {};
  };

  struct Crease {
    float w;
    int a, b;
    Crease() : w(0), a(-1), b(-1) {};
    Crease(float w, int a, int b) : w(w), a(a), b(b) {};
  };



  static inline bool operator < ( const Vertex& a, const Vertex& b ) {
    if (a.v  != b.v)  return a.v  < b.v;
    if (a.vn != b.vn) return a.vn < b.vn;
    if (a.vt != b.vt) return a.vt < b.vt;
    return false;
  }

  /*! Fill space at the end of the token with 0s. */
  static inline const char* trimEnd(const char* token) {
    size_t len = strlen(token);
    if (len == 0) return token;
    char* pe = (char*)(token + len - 1);
    while ((*pe == ' ' || *pe == '\t' || *pe == '\r') && pe >= token) *pe-- = 0;
    return token;
  }

  /*! Determine if character is a separator. */
  static inline bool isSep(const char c) {
    return (c == ' ') || (c == '\t');
  }

  /*! Parse separator. */
  static inline const char* parseSep(const char*& token) {
    size_t sep = strspn(token, " \t");
    if (!sep) THROW_RUNTIME_ERROR("separator expected");
    return token+=sep;
  }

  /*! Parse optional separator. */
  static inline const char* parseSepOpt(const char*& token) {
    return token+=strspn(token, " \t");
  }

  /*! Read float from a string. */
  static inline float getFloat(const char*& token) {
    token += strspn(token, " \t");
    float n = (float)atof(token);
    token += strcspn(token, " \t\r");
    return n;
  }

  /*! Read int from a string. */
  static inline int getInt(const char*& token) {
    token += strspn(token, " \t");
    int n = (float)atoi(token);
    token += strcspn(token, " \t\r");
    return n;
  }

  /*! Read Vec2f from a string. */
  static inline Vec2f getVec2f(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    return Vec2f(x,y);
  }

  /*! Read Vec3f from a string. */
  static inline Vec3f getVec3f(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    float z = getFloat(token);
    return Vec3f(x,y,z);
  }

  class OBJLoader
  {
  public:

    /*! Constructor. */
    OBJLoader(const FileName& fileName, const AffineSpace3f& space, OBJScene& mesh, const bool
              subdivMode);

    /*! Destruction */
    ~OBJLoader();
 
    /*! Public methods. */
    void loadMTL(const FileName& fileName);

  private:

    /*! file to load */
    FileName path;

    /*! output model */
    OBJScene& model;
    
    /*! load only quads and ignore triangles */
    bool subdivMode;

    /*! Geometry buffer. */
    avector<Vec3fa> v;
    avector<Vec3fa> vn;
    std::vector<Vec2f> vt;
    std::vector<Crease> ec;

    std::vector<int> vc;    
    std::vector<float> vcw; 

    std::vector<std::vector<Vertex> > curGroup;
    AffineSpace3f space;

    /*! Material handling. */
    int curMaterial;
    std::map<std::string, int> material;

    /*! Internal methods. */
    int fix_v (int index);
    int fix_vt(int index);
    int fix_vn(int index);
    void flushFaceGroup();
    Vertex getInt3(const char*& token);
    uint32_t getVertex(std::map<Vertex,uint32_t>& vertexMap, OBJScene::Mesh* mesh, const Vertex& i);
  };

  OBJLoader::OBJLoader(const FileName &fileName, const AffineSpace3f& space, OBJScene& mesh, const bool subdivMode) 
    : path(fileName.path()), model(mesh), space(space), subdivMode(subdivMode)
  {
    /* open file */
    std::ifstream cin;
    cin.open(fileName.c_str());
    if (!cin.is_open()) {
      THROW_RUNTIME_ERROR("cannot open " + fileName.str());
      return;
    }

    /* generate default material */
    model.materials.push_back(OBJScene::OBJMaterial());
    curMaterial = 0;

    char line[10000];
    memset(line, 0, sizeof(line));

    while (cin.peek() != -1)
    {
      /* load next multiline */
      char* pline = line;
      while (true) {
        cin.getline(pline, sizeof(line) - (pline - line) - 16, '\n');
        ssize_t last = strlen(pline) - 1;
        if (last < 0 || pline[last] != '\\') break;
        pline += last;
        *pline++ = ' ';
      }

      const char* token = trimEnd(line + strspn(line, " \t"));
      if (token[0] == 0) continue;

      /*! parse position */
      if (token[0] == 'v' && isSep(token[1])) { 
        const Vec3f p = xfmPoint(space,getVec3f(token += 2));
        v.push_back(p); continue;
      }

      /* parse normal */
      if (token[0] == 'v' && token[1] == 'n' && isSep(token[2])) { 
        const Vec3f n = xfmVector(space,getVec3f(token += 3));
        vn.push_back(n); 
        continue; 
      }

      /* parse texcoord */
      if (token[0] == 'v' && token[1] == 't' && isSep(token[2])) { vt.push_back(getVec2f(token += 3)); continue; }

      /*! parse face */
      if (token[0] == 'f' && isSep(token[1]))
      {
        parseSep(token += 1);

        std::vector<Vertex> face;
        while (token[0]) {
	  Vertex vtx = getInt3(token);
          face.push_back(vtx);
          parseSepOpt(token);
        }
        curGroup.push_back(face);
        continue;
      }

      /*! parse edge crease */
      if (token[0] == 'e' && token[1] == 'c' && isSep(token[2]))
      {
	parseSep(token += 2);
	float w = getFloat(token);
	parseSepOpt(token);
	int a = fix_v(getInt(token));
	parseSepOpt(token);
	int b = fix_v(getInt(token));
	parseSepOpt(token);
	ec.push_back(Crease(w, a, b));

	// HACK: EDGE CREASES FORCE VERTEX CREASES
#if 1
	vc.push_back(a);
	vc.push_back(b);
	vcw.push_back(w);
	vcw.push_back(w);
#endif
	continue;
      }

      /*! use material */
      if (!strncmp(token, "usemtl", 6) && isSep(token[6]))
      {
        flushFaceGroup();
        std::string name(parseSep(token += 6));
        if (material.find(name) == material.end()) curMaterial = 0;
        else curMaterial = material[name];
        continue;
      }

      /* load material library */
      if (!strncmp(token, "mtllib", 6) && isSep(token[6])) {
        loadMTL(path + std::string(parseSep(token += 6)));
        continue;
      }

      // ignore unknown stuff
    }
    flushFaceGroup();

    cin.close();
  }

  OBJLoader::~OBJLoader() {
  }
  
  /* load material file */
  void OBJLoader::loadMTL(const FileName &fileName)
  {
    std::ifstream cin;
    cin.open(fileName.c_str());
    if (!cin.is_open()) {
      std::cerr << "cannot open " << fileName.str() << std::endl;
      return;
    }

    char line[10000];
    memset(line, 0, sizeof(line));

    int cur = 0;
    while (cin.peek() != -1)
    {
      /* load next multiline */
      char* pline = line;
      while (true) {
        cin.getline(pline, sizeof(line) - (pline - line) - 16, '\n');
        ssize_t last = strlen(pline) - 1;
        if (last < 0 || pline[last] != '\\') break;
        pline += last;
        *pline++ = ' ';
      }
      const char* token = trimEnd(line + strspn(line, " \t"));

      if (token[0] == 0  ) continue; // ignore empty lines
      if (token[0] == '#') continue; // ignore comments

      if (!strncmp(token, "newmtl", 6)) {
        parseSep(token+=6);
        std::string name(token);
        material[name] = cur = model.materials.size();
        model.materials.push_back(OBJScene::OBJMaterial());
        continue;
      }

      if (!cur) THROW_RUNTIME_ERROR("invalid material file: newmtl expected first");

      if (!strncmp(token, "illum", 5)) { parseSep(token += 5);  continue; }

      if (!strncmp(token, "d",  1)) { parseSep(token += 1);  model.materials[cur].obj().d  = getFloat(token); continue; }
      if (!strncmp(token, "Ns", 2)) { parseSep(token += 2);  model.materials[cur].obj().Ns = getFloat(token); continue; }
      if (!strncmp(token, "Ni", 2)) { parseSep(token += 2);  model.materials[cur].obj().Ni = getFloat(token); continue; }

      if (!strncmp(token, "Ka_map", 6)) { continue; }
      if (!strncmp(token, "Kd_map", 6) || !strncmp(token, "map_Kd", 6)) {
        parseSep(token += 6);
        model.materials[cur].obj().map_Kd = loadTexture(path + FileName(token));
        continue;
      }


      if (!strncmp(token, "Ks_map", 6)) { continue; }
      if (!strncmp(token, "Tf_map", 6)) { continue; }
      if (!strncmp(token, "Displ_map", 9) || !strncmp(token, "map_Displ", 9)) {
        parseSep(token += 9);
        model.materials[cur].obj().map_Displ = loadTexture(path + FileName(token));
        continue;
      }
      
      if (!strncmp(token, "Ka", 2)) { parseSep(token += 2);  model.materials[cur].obj().Ka = getVec3f(token); continue; }
      if (!strncmp(token, "Kd", 2)) { parseSep(token += 2);  model.materials[cur].obj().Kd = getVec3f(token); 
#if 0
	if (model.materials[cur].obj().Kd == Vec3f(0,0,0))
	  model.materials[cur].obj().Kd = Vec3f(1,1,1);
#endif
	continue; 
      }
      if (!strncmp(token, "Ks", 2)) { parseSep(token += 2);  model.materials[cur].obj().Ks = getVec3f(token); continue; }
      if (!strncmp(token, "Tf", 2)) { parseSep(token += 2);  model.materials[cur].obj().Tf = getVec3f(token); continue; }
    }
    cin.close();
  }

  /*! handles relative indices and starts indexing from 0 */
  int OBJLoader::fix_v (int index) { return (index > 0 ? index - 1 : (index == 0 ? 0 : (int) v .size() + index)); }
  int OBJLoader::fix_vt(int index) { return (index > 0 ? index - 1 : (index == 0 ? 0 : (int) vt.size() + index)); }
  int OBJLoader::fix_vn(int index) { return (index > 0 ? index - 1 : (index == 0 ? 0 : (int) vn.size() + index)); }

  /*! Parse differently formated triplets like: n0, n0/n1/n2, n0//n2, n0/n1.          */
  /*! All indices are converted to C-style (from 0). Missing entries are assigned -1. */
  Vertex OBJLoader::getInt3(const char*& token)
  {
    Vertex v(-1);
    v.v = fix_v(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;

    // it is i//n
    if (token[0] == '/') {
      token++;
      v.vn = fix_vn(atoi(token));
      token += strcspn(token, " \t\r");
      return(v);
    }

    // it is i/t/n or i/t
    v.vt = fix_vt(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;

    // it is i/t/n
    v.vn = fix_vn(atoi(token));
    token += strcspn(token, " \t\r");
    return(v);
  }

  uint32_t OBJLoader::getVertex(std::map<Vertex,uint32_t>& vertexMap, OBJScene::Mesh* mesh, const Vertex& i)
  {
    const std::map<Vertex, uint32_t>::iterator& entry = vertexMap.find(i);
    if (entry != vertexMap.end()) return(entry->second);
    mesh->v.push_back(Vec3fa(v[i.v].x,v[i.v].y,v[i.v].z));
    if (i.vn >= 0) mesh->vn.push_back(vn[i.vn]);
    if (i.vt >= 0) mesh->vt.push_back(vt[i.vt]);
    return(vertexMap[i] = int(mesh->v.size()) - 1);
  }

  /*! end current facegroup and append to mesh */
  void OBJLoader::flushFaceGroup()
  {
    if (curGroup.empty()) return;
    
    if (subdivMode)
      {
	OBJScene::SubdivMesh* mesh = new OBJScene::SubdivMesh;
	model.subdiv.push_back(mesh);
#if 0
	PRINT(curGroup.size());
	PRINT(v.size());
	PRINT(vn.size());
	PRINT(vt.size());
#endif	

	for (size_t i=0;i<v.size();i++)  mesh->positions.push_back(v[i]);
	for (size_t i=0;i<vn.size();i++) mesh->normals.push_back(vn[i]);
	for (size_t i=0;i<vt.size();i++) mesh->texcoords.push_back(vt[i]);
	
#if 1
	for (size_t i=0;i<ec.size();++i) {
	  if (ec[i].a < v.size() && ec[i].b < v.size())
	    mesh->edge_creases.push_back(Vec2i(ec[i].a, ec[i].b));
	  //mesh->edge_crease_weights.push_back(ec[i].w);
	  mesh->edge_crease_weights.push_back(pos_inf);

	}
#endif

	for (size_t i=0;i<vc.size();++i) 
	  mesh->vertex_creases.push_back(vc[i]);
	for (size_t i=0;i<vcw.size();++i) 
	  mesh->vertex_crease_weights.push_back(pos_inf);	  
	  //mesh->vertex_crease_weights.push_back(vcw[i]);	  

	for (size_t j=0; j < curGroup.size(); j++)
	  {
	    /* iterate over all faces */
	    const std::vector<Vertex>& face = curGroup[j];

#if FORCE_ONLY_QUADS == 1
	    if ( face.size() == 4)
#endif
	      {
		for (size_t i=0;i<face.size();i++)
		  {
		    mesh->position_indices.push_back(face[i].v);
		    if (face[i].vt != -1)
		      {
			assert( face[i].vt < vt.size() );
			mesh->texcoord_indices.push_back(face[i].vt);
		      }
		  }
		if (mesh->texcoord_indices.size())
		  {
		    //PRINT( mesh->texcoord_indices.size() );
		    //PRINT( mesh->position_indices.size() );

		    //assert( mesh->texcoord_indices.size() == mesh->position_indices.size() );
		  }
		mesh->verticesPerFace.push_back(face.size());
	      }
            
	    mesh->materialID = curMaterial;	
	  }
      }
    else
      {
	OBJScene::Mesh* mesh = new OBJScene::Mesh;
	model.meshes.push_back(mesh);

	// merge three indices into one
	std::map<Vertex, uint32_t> vertexMap;
	for (size_t j=0; j < curGroup.size(); j++)
	  {
	    /* iterate over all faces */
	    const std::vector<Vertex>& face = curGroup[j];

	    /* triangulate the face with a triangle fan */
	    Vertex i0 = face[0], i1 = Vertex(-1), i2 = face[1];
	    for (size_t k=2; k < face.size(); k++) {
	      i1 = i2; i2 = face[k];
	      uint32_t v0,v1,v2;
	      v0 = getVertex(vertexMap, mesh, i0);
	      v1 = getVertex(vertexMap, mesh, i1);
	      v2 = getVertex(vertexMap, mesh, i2);
	      assert(v0 < mesh->v.size());
	      assert(v1 < mesh->v.size());
	      assert(v2 < mesh->v.size());
	      mesh->triangles.push_back(OBJScene::Triangle(v0,v1,v2,curMaterial));
	    }
	  }

	mesh->meshMaterialID = curMaterial;	
      }
    curGroup.clear();
  }

  void loadOBJ(const FileName& fileName, const AffineSpace3f& space, OBJScene& mesh_o, const bool subdivMode) {
    OBJLoader loader(fileName,space,mesh_o,subdivMode); 
  }

  void OBJScene::Mesh::set_motion_blur(const Mesh* other)
  {
    if (v.size() != other->v.size())
      THROW_RUNTIME_ERROR("incompatible geometry");

    bool different = false;
    for (size_t i=0; i<v.size(); i++) 
      different |= v[i] != other->v[i];

    if (different)
      v2 = other->v;
  }

  void OBJScene::HairSet::set_motion_blur(const HairSet* other)
  {
    if (v.size() != other->v.size())
      THROW_RUNTIME_ERROR("incompatible geometry");

    bool different = false;
    for (size_t i=0; i<v.size(); i++) 
      different |= v[i] != other->v[i];

    if (different)
      v2 = other->v;
  }

  void OBJScene::set_motion_blur(OBJScene& other)
  {
    if (meshes.size() != other.meshes.size())
      THROW_RUNTIME_ERROR("incompatible geometry");
    
    for (size_t i=0; i<meshes.size(); i++) 
      meshes[i]->set_motion_blur(other.meshes[i]);

    if (hairsets.size() != other.hairsets.size())
      THROW_RUNTIME_ERROR("incompatible geometry");

    for (size_t i=0; i<hairsets.size(); i++) 
      hairsets[i]->set_motion_blur(other.hairsets[i]);
  }
}

