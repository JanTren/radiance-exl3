# kolibri1_exl3 -- the ("kolibri1", "exl3") architecture plugin. Built inside this repository's
# top-level project, against an install of the radiance-kolibri1 fork of radiance 1.7.0 (branch
# kolibri1-1.7): its headers carry rad_kolibri1_chat.h, and its kernel libraries the ops Kolibri adds
# (router_sigmoid_topk, qk_norm_rope at rotary_dim 0, windowed attention at gqa 12). Stock 1.7.0
# fails here, at the missing header, rather than at declare.
if(NOT EXISTS "${RADIANCE_INCLUDE_DIR}/arch/rad_kolibri1_chat.h")
  message(STATUS "kolibri1_exl3: SKIPPED -- ${RADIANCE_INCLUDE_DIR} has no arch/rad_kolibri1_chat.h "
                 "(build against the radiance-kolibri1 fork's install to get this plugin)")
  return()
endif()

rad_add_plugin(kolibri1_exl3 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kolibri1_exl3.cpp)
# The MoeEXL3 block's inline members are compiled here and in qwen4exp_exl3.so; hidden, each .so
# keeps its own copy however a process loads the two.
target_compile_options(kolibri1_exl3 PRIVATE -fvisibility-inlines-hidden)
