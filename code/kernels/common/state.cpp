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

#include "state.h"
#include "../../common/lexers/streamfilters.h"

namespace embree
{
  /* error flag */
  tls_t State::g_error = nullptr; // FIXME: use thread local
  std::vector<RTCError*> State::g_errors; // FIXME: use thread local
  MutexSys State::g_errors_mutex;

  State State::state;

  State::State () {
    clear();
  }
  
  void State::clear()
  {
    tri_accel = "default";
    tri_builder = "default";
    tri_traverser = "default";
    tri_builder_replication_factor = 2.0f;
    
    tri_accel_mb = "default";
    tri_builder_mb = "default";
    tri_traverser_mb = "default";
    
    hair_accel = "default";
    hair_builder = "default";
    hair_traverser = "default";
    hair_builder_replication_factor = 3.0f;
    memory_preallocation_factor     = 1.0f; 
    tessellation_cache_size         = 0;
    subdiv_accel = "default";
    
    scene_flags = -1;
    verbose = 0;
    g_numThreads = 0;
    benchmark = 0;
    regression_testing = 0;

    {
      Lock<MutexSys> lock(g_errors_mutex);
      if (g_error == nullptr) 
        g_error = createTls();
    }
    g_error_function = nullptr;
    g_memory_monitor_function = nullptr;

    //Lock<MutexSys> lock(g_errors_mutex);
    //  for (size_t i=0; i<g_errors.size(); i++)
    //    delete g_errors[i];
    //  destroyTls(g_error);
    //  g_errors.clear();
  }

  const char* symbols[3] = { "=", ",", "|" };

   bool State::parseFile(const FileName& fileName)
  {
    Ref<Stream<int> > file;
    try {
      file = new FileStream(fileName);
    } catch (const std::runtime_error&) {
      return false;
    }

    std::vector<std::string> syms;
	  for (size_t i=0; i<sizeof(symbols)/sizeof(void*); i++) 
      syms.push_back(symbols[i]);

    Ref<TokenStream> cin = new TokenStream(new LineCommentFilter(file,"#"),
                                           TokenStream::alpha+TokenStream::ALPHA+TokenStream::numbers+"_.",
                                           TokenStream::separators,syms);
    parse(cin);
    return true;
  }

  void State::parseString(const char* cfg)
  {
    if (cfg == nullptr) return;

    std::vector<std::string> syms;
    for (size_t i=0; i<sizeof(symbols)/sizeof(void*); i++) 
      syms.push_back(symbols[i]);
    
    Ref<TokenStream> cin = new TokenStream(new StrStream(cfg),
                                           TokenStream::alpha+TokenStream::ALPHA+TokenStream::numbers+"_.",
                                           TokenStream::separators,syms);
    parse(cin);
  }

  void State::parse(Ref<TokenStream> cin)
  {
    /* parse until end of stream */
    while (cin->peek() != Token::Eof())
    {
      const Token tok = cin->get();

      if (tok == Token::Id("threads") && cin->trySymbol("=")) 
        g_numThreads = cin->get().Int();
      
      else if (tok == Token::Id("isa") && cin->trySymbol("=")) 
      {
        std::string isa = cin->get().Identifier();
        if      (isa == "sse" ) setCPUFeatures(SSE);
        else if (isa == "sse2") setCPUFeatures(SSE2);
        else if (isa == "sse3") setCPUFeatures(SSE3);
        else if (isa == "ssse3") setCPUFeatures(SSSE3);
        else if (isa == "sse41") setCPUFeatures(SSE41);
        else if (isa == "sse4.1") setCPUFeatures(SSE41);
        else if (isa == "sse42") setCPUFeatures(SSE42);
        else if (isa == "sse4.2") setCPUFeatures(SSE42);
        else if (isa == "avx") setCPUFeatures(AVX);
        else if (isa == "int8") setCPUFeatures(AVXI);
        else if (isa == "avx2") setCPUFeatures(AVX2);
      }

      else if (tok == Token::Id("float_exceptions") && cin->trySymbol("=")) 
        float_exceptions = cin->get().Int();

      else if ((tok == Token::Id("tri_accel") || tok == Token::Id("accel")) && cin->trySymbol("="))
        tri_accel = cin->get().Identifier();
      else if ((tok == Token::Id("tri_builder") || tok == Token::Id("builder")) && cin->trySymbol("="))
        tri_builder = cin->get().Identifier();
      else if ((tok == Token::Id("tri_traverser") || tok == Token::Id("traverser")) && cin->trySymbol("="))
        tri_traverser = cin->get().Identifier();
      else if (tok == Token::Id("tri_builder_replication_factor") && cin->trySymbol("="))
        tri_builder_replication_factor = cin->get().Int();
      
      else if ((tok == Token::Id("tri_accel_mb") || tok == Token::Id("accel_mb")) && cin->trySymbol("="))
        tri_accel_mb = cin->get().Identifier();
      else if ((tok == Token::Id("tri_builder_mb") || tok == Token::Id("builder_mb")) && cin->trySymbol("="))
        tri_builder_mb = cin->get().Identifier();
      else if ((tok == Token::Id("tri_traverser_mb") || tok == Token::Id("traverser_mb")) && cin->trySymbol("="))
        tri_traverser_mb = cin->get().Identifier();
      
      else if (tok == Token::Id("hair_accel") && cin->trySymbol("="))
        hair_accel = cin->get().Identifier();
      else if (tok == Token::Id("hair_builder") && cin->trySymbol("="))
        hair_builder = cin->get().Identifier();
      else if (tok == Token::Id("hair_traverser") && cin->trySymbol("="))
        hair_traverser = cin->get().Identifier();
      else if (tok == Token::Id("hair_builder_replication_factor") && cin->trySymbol("="))
        hair_builder_replication_factor = cin->get().Int();
      
      else if (tok == Token::Id("subdiv_accel") && cin->trySymbol("="))
        subdiv_accel = cin->get().Identifier();
      
      else if (tok == Token::Id("verbose") && cin->trySymbol("="))
        verbose = cin->get().Int();
      else if (tok == Token::Id("benchmark") && cin->trySymbol("="))
        benchmark = cin->get().Int();
      
      else if (tok == Token::Id("flags")) {
        scene_flags = 0;
        if (cin->trySymbol("=")) {
          do {
            Token flag = cin->get();
            if      (flag == Token::Id("static") ) scene_flags |= RTC_SCENE_STATIC;
            else if (flag == Token::Id("dynamic")) scene_flags |= RTC_SCENE_DYNAMIC;
            else if (flag == Token::Id("compact")) scene_flags |= RTC_SCENE_COMPACT;
            else if (flag == Token::Id("coherent")) scene_flags |= RTC_SCENE_COHERENT;
            else if (flag == Token::Id("incoherent")) scene_flags |= RTC_SCENE_INCOHERENT;
            else if (flag == Token::Id("high_quality")) scene_flags |= RTC_SCENE_HIGH_QUALITY;
            else if (flag == Token::Id("robust")) scene_flags |= RTC_SCENE_ROBUST;
          } while (cin->trySymbol("|"));
        }
      }
      else if (tok == Token::Id("memory_preallocation_factor") && cin->trySymbol("=")) 
        memory_preallocation_factor = cin->get().Float();
      
      else if (tok == Token::Id("regression") && cin->trySymbol("=")) 
        regression_testing = cin->get().Int();
      
      else if (tok == Token::Id("tessellation_cache_size") && cin->trySymbol("="))
        tessellation_cache_size = cin->get().Float() * 1024 * 1024;

      cin->trySymbol(","); // optional , separator
    }
  }

  RTCError* State::error() 
  {
    RTCError* stored_error = (RTCError*) getTls(g_error);
    if (stored_error == nullptr) {
      Lock<MutexSys> lock(g_errors_mutex);
      stored_error = new RTCError(RTC_NO_ERROR);
      g_errors.push_back(stored_error);
      setTls(g_error,stored_error);
    }
    return stored_error;
  }

  bool State::verbosity(int N) {
    return N <= verbose;
  }
  
  void State::print()
  {
    std::cout << "general:" << std::endl;
    std::cout << "  build threads = " << g_numThreads << std::endl;
    std::cout << "  verbosity     = " << verbose << std::endl;
    
    std::cout << "triangles:" << std::endl;
    std::cout << "  accel         = " << tri_accel << std::endl;
    std::cout << "  builder       = " << tri_builder << std::endl;
    std::cout << "  traverser     = " << tri_traverser << std::endl;
    std::cout << "  replications  = " << tri_builder_replication_factor << std::endl;
    
    std::cout << "motion blur triangles:" << std::endl;
    std::cout << "  accel         = " << tri_accel_mb << std::endl;
    std::cout << "  builder       = " << tri_builder_mb << std::endl;
    std::cout << "  traverser     = " << tri_traverser_mb << std::endl;
    
    std::cout << "hair:" << std::endl;
    std::cout << "  accel         = " << hair_accel << std::endl;
    std::cout << "  builder       = " << hair_builder << std::endl;
    std::cout << "  traverser     = " << hair_traverser << std::endl;
    std::cout << "  replications  = " << hair_builder_replication_factor << std::endl;
    
    std::cout << "subdivision surfaces:" << std::endl;
    std::cout << "  accel         = " << subdiv_accel << std::endl;
    
#if defined(__MIC__)
    std::cout << "memory allocation:" << std::endl;
    std::cout << "  preallocation_factor  = " << memory_preallocation_factor << std::endl;
#endif
  }
}
