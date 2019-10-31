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

//#include "../../sys/sysinfo.h"
#include "../common/tutorial/scene.h"
#include "../common/transport/transport_host.h"
#include "../common/tutorial/obj_loader.h"
#include <optixu/optixpp.h>

#define PTX_PATH(f) "tutorials/pathtracer/pathtracer_optix_generated_" f ".cu.ptx"

/*extern "C" int64_t get_tsc() {
  return __rdtsc();
  }*/

namespace embree
{
  void set_scene_keyframes(OBJScene** in, size_t numKeyFrames)
  {
  }

  const FileName ptxfile = FileName::executableFolder() + FileName(PTX_PATH("pathtracer_optix"));

  optixu::Context context;	 /* optix context handle */
  optixu::Material material;
  optixu::Buffer raycounter;
  optixu::Acceleration accel;	 /* acceleration structure for readback */
  
  /* framebuffer */
  int g_width = -1;
  int g_height = -1;
  optixu::Buffer frameBuffer;	 /* handle to framebuffer */

  //std::string strAccel = "NoAccel"; std::string strTraverser = "NoAccel";
  //std::string strAccel = "Bvh"; std::string strTraverser = "Bvh";
  //std::string strAccel = "Lbvh"; std::string strTraverser = "Bvh";
  //std::string strAccel = "MedianBvh"; std::string strTraverser = "Bvh";
  //std::string strAccel = "Sbvh"; std::string strTraverser = "Bvh";
  //std::string strAccel = "TriangleKdTree"; std::string strTraverser = "KdTree";
  std::string strAccel = "Trbvh"; std::string strTraverser = "Bvh";
  
  void init(const char* cfg) try
  {
    /* enable RTX */
    int rtx = 1;
    RTresult rtxResult;
    if ((rtxResult = rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx)) != RT_SUCCESS)
       std::cout << "using RTX" << std::endl;
    else
      std::cout << "not using RTX" << std::endl;
   
    /* setup context */
    context = optixu::ContextObj::create();
    std::vector<int> devices = {0}; // use only the first CUDA device!
    context->setDevices(devices.begin(), devices.end());
    context->setRayTypeCount(2);
    context->setEntryPointCount(1);
    
    /* set default values */
    context["P"]->setFloat(0,0,0);
    context["U"]->setFloat(0,0,0);
    context["V"]->setFloat(0,0,0);
    context["W"]->setFloat(0,0,0);
    context["frameBuffer"]->setBuffer(context->createBuffer(RT_BUFFER_OUTPUT,RT_FORMAT_FLOAT,1,1));
    
    raycounter = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_INT,1);
    context["raycounter"]->setBuffer(raycounter);
    
    /* set ray generation program */
    context->setRayGenerationProgram(0,context->createProgramFromPTXFile(ptxfile.c_str(), "render_primary"));
    context->setMissProgram( 0, context->createProgramFromPTXFile( ptxfile.c_str(), "miss_program" ));
    context->setMissProgram( 1, context->createProgramFromPTXFile( ptxfile.c_str(), "miss_program" ));
  } 
  catch( optixu::Exception& e ){
    std::cout << "Error: " << e.getErrorString().c_str()<< std::endl;
    exit(1);
  }

  void key_pressed (int32_t key) {
  }

  void resize(int32_t width, int32_t height)
  {
    if (width == g_width && height == g_height)
      return;

    g_width = width; g_height = height;
    frameBuffer = context->createBuffer(RT_BUFFER_OUTPUT,RT_FORMAT_UNSIGNED_BYTE4,width,height);
    context["frameBuffer"]->setBuffer(frameBuffer);
  }

  void set_scene (OBJScene* scene) try
  {
    context["g_accu_count"]->setInt(0);

    /* create material buffer */
    //optixu::Buffer material_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT4,5*scene->materials.size());
    optixu::Buffer material_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_USER);
    material_buffer->setElementSize(sizeof(OBJScene::Material));
    material_buffer->setSize(scene->materials.size());
    context["materials"]->setBuffer(material_buffer);
    OBJScene::Material* material_data = (OBJScene::Material*) material_buffer->map();
    for (unsigned i=0; i<scene->materials.size(); i++) material_data[i] = scene->materials[i];
    material_buffer->unmap();

     /* create ambientLights buffer */
    context["numAmbientLights"]->setInt(scene->ambientLights.size());
    optixu::Buffer ambientLights_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_USER);
    ambientLights_buffer->setElementSize(sizeof(OBJScene::AmbientLight));
    ambientLights_buffer->setSize(scene->ambientLights.size());
    context["ambientLights"]->setBuffer(ambientLights_buffer);
    OBJScene::AmbientLight* ambientLights_data = (OBJScene::AmbientLight*) ambientLights_buffer->map();
    for (unsigned i=0; i<scene->ambientLights.size(); i++) ambientLights_data[i] = scene->ambientLights[i];
    ambientLights_buffer->unmap();

    /* create pointLights buffer */
    context["numPointLights"]->setInt(scene->pointLights.size());
    optixu::Buffer pointLights_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_USER);
    pointLights_buffer->setElementSize(sizeof(OBJScene::PointLight));
    pointLights_buffer->setSize(scene->pointLights.size());
    context["pointLights"]->setBuffer(pointLights_buffer);
    OBJScene::PointLight* pointLights_data = (OBJScene::PointLight*) pointLights_buffer->map();
    for (unsigned i=0; i<scene->pointLights.size(); i++) pointLights_data[i] = scene->pointLights[i];
    pointLights_buffer->unmap();

    /* create dirLights buffer */
    context["numDirectionalLights"]->setInt(scene->directionalLights.size());
    optixu::Buffer directionalLights_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_USER);
    directionalLights_buffer->setElementSize(sizeof(OBJScene::DirectionalLight));
    directionalLights_buffer->setSize(scene->directionalLights.size());
    context["dirLights"]->setBuffer(directionalLights_buffer);
    OBJScene::DirectionalLight* dirLights_data = (OBJScene::DirectionalLight*) directionalLights_buffer->map();
    for (unsigned i=0; i<scene->directionalLights.size(); i++) dirLights_data[i] = scene->directionalLights[i];
    directionalLights_buffer->unmap();

    /* create distantLights buffer */
    context["numDistantLights"]->setInt(scene->distantLights.size());
    optixu::Buffer distantLights_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_USER);
    distantLights_buffer->setElementSize(sizeof(OBJScene::DistantLight));
    distantLights_buffer->setSize(scene->distantLights.size());
    context["distantLights"]->setBuffer(distantLights_buffer);
    OBJScene::DistantLight* distantLights_data = (OBJScene::DistantLight*) distantLights_buffer->map();
    for (unsigned i=0; i<scene->distantLights.size(); i++) distantLights_data[i] = scene->distantLights[i];
    distantLights_buffer->unmap();
    
    std::vector<Vec3f> positions;
    std::vector<OBJScene::Triangle> triangles;
    
    for (size_t i=0; i<scene->meshes.size(); i++)
    {
      OBJScene::Mesh* mesh = scene->meshes[i];
      int base = positions.size();

      for (size_t j=0; j<mesh->v.size(); j++)
	positions.push_back(mesh->v[j]);

      for (size_t j=0; j<mesh->triangles.size(); j++) 
      {
	int v0 = mesh->triangles[j].v0;
	int v1 = mesh->triangles[j].v1;
	int v2 = mesh->triangles[j].v2;
	int materialID = mesh->triangles[j].materialID;
	triangles.push_back(OBJScene::Triangle(v0+base,v1+base,v2+base,materialID));
      }
    }

    std::cout << " triangles = " << triangles.size() << ", vertices = " << positions.size() << " [DONE]" << std::endl;

    /* create geometry */
    optixu::Geometry mesh = context->createGeometry();
    mesh->setPrimitiveCount(triangles.size());
    mesh->setBoundingBoxProgram(context->createProgramFromPTXFile(ptxfile.c_str(), "triangle_bounds"));
    mesh->setIntersectionProgram(context->createProgramFromPTXFile(ptxfile.c_str(), "triangle_intersect"));

    optixu::Buffer triangle_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT3,triangles.size());
    optixu::Buffer materialID_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT,triangles.size());
    optixu::Buffer position_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,positions.size());

    context["triangles"]->setBuffer(triangle_buffer);
    context["trimaterialID"]->setBuffer(materialID_buffer);
    context["positions"]->setBuffer(position_buffer);
    
    /* copy data into mesh buffers */
    Vec3i* triangle_data = (Vec3i*) triangle_buffer->map();
    for (unsigned i=0; i<triangles.size(); i++) triangle_data[i] = Vec3i(triangles[i].v0,triangles[i].v1,triangles[i].v2);
    triangle_buffer->unmap();

    int* materialID_data = (int*) materialID_buffer->map();
    for (unsigned i=0; i<triangles.size(); i++) materialID_data[i] = triangles[i].materialID;
    materialID_buffer->unmap();
    
    Vec3f* position_data = (Vec3f*) position_buffer->map();
    for (unsigned i=0; i<positions.size(); i++) position_data[i] = positions[i];
    position_buffer->unmap();

    /* create material */
    material = context->createMaterial();
    material->setClosestHitProgram( 0, context->createProgramFromPTXFile( ptxfile.c_str(), "closest_hit_program" ));
    material->setAnyHitProgram( 1, context->createProgramFromPTXFile( ptxfile.c_str(), "any_hit_program" ));

    /* instantiate geometry with material */
    optixu::GeometryInstance instance = context->createGeometryInstance();
    instance->setGeometry(mesh);
    instance->setMaterialCount(1);
    instance->setMaterial(0,material);
    
    /* create acceleration structure */
    accel = context->createAcceleration(strAccel.c_str(),strTraverser.c_str());
    accel->setProperty("index_buffer_name","triangles");
    accel->setProperty("vertex_buffer_name","positions");
    
    /* create group to hold instance transform */
    optixu::GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount(1);
    geometrygroup->setChild(0,instance);
    geometrygroup->setAcceleration(accel);
    context["g_scene"]->set(geometrygroup);
  }
  catch( optixu::Exception& e ){
    std::cout << "Error: " << e.getErrorString().c_str()<< std::endl;
    exit(1);
  }

  bool pick(const float x, const float y, const Vec3fa& vx, const Vec3fa& vy, const Vec3fa& vz, const Vec3fa& p, Vec3fa& hitPos) {
    return false;
  }

  void render(const float time, const Vec3fa& vx, const Vec3fa& vy, const Vec3fa& vz, const Vec3fa& p) try
  {
    if (raycounter) {
      int* counter = (int*) raycounter->map();
      counter[0] = 0;
      raycounter->unmap();
    }
    
    /* set camera */
    context["P"]->setFloat(p .x, p .y, p .z);
    context["U"]->setFloat(vx.x, vx.y, vx.z);
    context["V"]->setFloat(vy.x, vy.y, vy.z);
    context["W"]->setFloat(vz.x, vz.y, vz.z);

    /* run context */
    //double t0 = getSeconds();
    context->launch(0,g_width,g_height);
    double dt = 0.0f; //getSeconds() - t0;

    /* print frame rate */
    /*std::ostringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream.precision(2);
    stream << "optix render: ";
    stream << 1.0f/dt << " fps, ";
    stream << dt*1000.0f << " ms, ";
    stream << g_width << "x" << g_height << " pixels";
    std::cout << stream.str() << std::endl;*/
    
    if (raycounter) {
      int* counter = (int*) raycounter->map();
      if (counter[0]) std::cout << "number of rays = " << float(counter[0])*1E-6 << "M" << std::endl;
      raycounter->unmap();
    }
  }
  catch( optixu::Exception& e ){
    std::cout << "Error: " << e.getErrorString().c_str()<< std::endl;
    exit(1);
  }

  int* map () {
    return (int*) frameBuffer->map();
  }
  
  void unmap () {
    frameBuffer->unmap();
  }

  void cleanup() {
    context->destroy();
  }
}


