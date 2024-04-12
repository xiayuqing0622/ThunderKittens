/* ----------  TEST INCLUSION MACROS  ---------- */

/* -----  DEPTH 0 MACROS  ----- */

// Catchall macros
#ifdef TEST_ALL
#define TEST_ALL_WARP
#define TEST_ALL_WARPGROUP
#define TEST_ALL_GROUP
#endif

/* -----  DEPTH 1 MACROS  ----- */

// Warp macros
#ifdef TEST_ALL_WARP
#define TEST_ALL_WARP_MEMORY
#define TEST_ALL_WARP_REGISTER
#define TEST_ALL_WARP_SHARED
#endif

// Group macros
#ifdef TEST_ALL_GROUP
#define TEST_ALL_GROUP_MEMORY
#define TEST_ALL_GROUP_SHARED
#ifdef KITTENS_HOPPER  // only compile on H100
#define TEST_ALL_WARPGROUP_WGMMA // leaf
#endif
#endif

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#ifdef TEST_ALL_WARP_MEMORY
#define TEST_ALL_WARP_MEMORY_TILE
#define TEST_ALL_WARP_MEMORY_VEC
#endif

#ifdef TEST_ALL_WARP_REGISTER
#define TEST_ALL_WARP_REGISTER_TILE
#define TEST_ALL_WARP_REGISTER_VEC
#endif

#ifdef TEST_ALL_WARP_SHARED
#define TEST_ALL_WARP_SHARED_TILE
#define TEST_ALL_WARP_SHARED_VEC
#endif

// Group macros

#ifdef TEST_ALL_GROUP_MEMORY
#define TEST_ALL_GROUP_MEMORY_TILE
#define TEST_ALL_GROUP_MEMORY_VEC
#endif

#ifdef TEST_ALL_GROUP_SHARED
#define TEST_ALL_GROUP_SHARED_TILE
#define TEST_ALL_GROUP_SHARED_VEC
#endif

/* -----  DEPTH 3 MACROS  ----- */

// Warp macros

#ifdef TEST_ALL_WARP_MEMORY_TILE
#define TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
#define TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
#define TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_WARP_MEMORY_TILE_TMA
#define TEST_WARP_MEMORY_TILE_DSMEM
#endif
#endif

#ifdef TEST_ALL_WARP_MEMORY_VEC
#define TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER
#define TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED
#define TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_WARP_MEMORY_VEC_TMA
#define TEST_WARP_MEMORY_VEC_DSMEM
#endif
#endif

#ifdef TEST_ALL_WARP_REGISTER_TILE
#define TEST_WARP_REGISTER_TILE_REDUCTIONS
#define TEST_WARP_REGISTER_TILE_MAPS
#define TEST_WARP_REGISTER_TILE_MMA
#define TEST_WARP_REGISTER_TILE_CONVERSIONS
#endif

#ifdef TEST_ALL_WARP_REGISTER_VEC
#define TEST_WARP_REGISTER_VEC_REDUCTIONS
#define TEST_WARP_REGISTER_VEC_MAPS
#define TEST_WARP_REGISTER_VEC_CONVERSIONS
#endif

#ifdef TEST_ALL_WARP_SHARED_TILE
#define TEST_WARP_SHARED_TILE_CONVERSIONS
#define TEST_WARP_SHARED_TILE_MAPS
#define TEST_WARP_SHARED_TILE_REDUCTIONS
#endif

#ifdef TEST_ALL_WARP_SHARED_VEC
#define TEST_WARP_SHARED_VEC_CONVERSIONS
#define TEST_WARP_SHARED_VEC_MAPS
#define TEST_WARP_SHARED_VEC_REDUCTIONS
#endif

// Group macros

#ifdef TEST_ALL_GROUP_MEMORY_TILE
#define TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER
#define TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
#define TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
#endif

#ifdef TEST_ALL_GROUP_MEMORY_VEC
#define TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER
#define TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
#define TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER
#endif

#ifdef TEST_ALL_GROUP_SHARED_TILE
#define TEST_GROUP_SHARED_TILE_REDUCTIONS
#define TEST_GROUP_SHARED_TILE_MAPS
#define TEST_GROUP_SHARED_TILE_CONVERSIONS
#endif

#ifdef TEST_ALL_GROUP_SHARED_VEC
#define TEST_GROUP_SHARED_VEC_MAPS
#define TEST_GROUP_SHARED_VEC_CONVERSIONS
#endif

// Now we need to go back up the tree and make sure all dependent flags are defined.

/* -----  DEPTH 3 MACROS  ----- */

// Warp macros

#if defined(TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER) || defined(TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER) || defined(TEST_WARP_MEMORY_TILE_TMA) || \
    defined(TEST_WARP_MEMORY_TILE_DSMEM)
#define TEST_WARP_MEMORY_TILE
#endif

#if defined(TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER) || defined(TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER) || defined(TEST_WARP_MEMORY_VEC_TMA) || \
    defined(TEST_WARP_MEMORY_VEC_DSMEM)
#define TEST_WARP_MEMORY_VEC
#endif

#if defined(TEST_WARP_REGISTER_TILE_REDUCTIONS) || defined(TEST_WARP_REGISTER_TILE_MAPS) || \
    defined(TEST_WARP_REGISTER_TILE_MMA) || defined(TEST_WARP_REGISTER_TILE_CONVERSIONS)
#define TEST_WARP_REGISTER_TILE
#endif

#if defined(TEST_WARP_REGISTER_VEC_REDUCTIONS) || defined(TEST_WARP_REGISTER_VEC_MAPS) || \
    defined(TEST_WARP_REGISTER_VEC_CONVERSIONS)
#define TEST_WARP_REGISTER_VEC
#endif

#if defined(TEST_WARP_SHARED_TILE_CONVERSIONS) || defined(TEST_WARP_SHARED_TILE_MAPS) || \
    defined(TEST_WARP_SHARED_TILE_REDUCTIONS)
#define TEST_WARP_SHARED_TILE
#endif

#if defined(TEST_WARP_SHARED_VEC_CONVERSIONS) || defined(TEST_WARP_SHARED_VEC_MAPS) || \
    defined(TEST_WARP_SHARED_VEC_REDUCTIONS)
#define TEST_WARP_SHARED_VEC
#endif

// Group macros

#if defined(TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER) || defined(TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER)
#define TEST_GROUP_MEMORY_TILE
#endif

#if defined(TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER) || defined(TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER)
#define TEST_GROUP_MEMORY_VEC
#endif

#if defined(TEST_GROUP_SHARED_TILE_CONVERSIONS) || defined(TEST_GROUP_SHARED_TILE_MAPS) || \
    defined(TEST_GROUP_SHARED_TILE_REDUCTIONS)
#define TEST_GROUP_SHARED_TILE
#endif

#if defined(TEST_GROUP_SHARED_VEC_CONVERSIONS) || defined(TEST_GROUP_SHARED_VEC_MAPS)
#define TEST_GROUP_SHARED_VEC
#endif

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#if defined(TEST_ALL_WARP_MEMORY_TILE) || defined(TEST_ALL_WARP_MEMORY_VEC)
#define TEST_WARP_MEMORY
#endif

#if defined(TEST_WARP_REGISTER_TILE) || defined(TEST_WARP_REGISTER_VEC)
#define TEST_WARP_REGISTER
#endif

#if defined(TEST_WARP_SHARED_TILE) || defined(TEST_WARP_SHARED_VEC)
#define TEST_WARP_SHARED
#endif

// Group macros

#if defined(TEST_ALL_GROUP_MEMORY_TILE) || defined(TEST_ALL_GROUP_MEMORY_VEC)
#define TEST_GROUP_MEMORY
#endif

#if defined(TEST_GROUP_SHARED_TILE) || defined(TEST_GROUP_SHARED_VEC)
#define TEST_GROUP_SHARED
#endif

/* -----  DEPTH 1 MACROS  ----- */

#if defined(TEST_WARP_MEMORY) || defined(TEST_WARP_REGISTER) || defined(TEST_WARP_SHARED)
#define TEST_WARP
#endif

#if defined(TEST_GROUP_MEMORY) || defined(TEST_GROUP_SHARED)
#define TEST_GROUP
#endif

/* ----------  TEST INTENSITY MACROS  ---------- */

// Intensity 1 is a cursory glance
#define TEST_INTENSITY_1 (1)
// Intensity 2 is to actually check
#define TEST_INTENSITY_2 (2)
// Intensity 3 is a thorough check
#define TEST_INTENSITY_3 (3)
// Intensity 4 is for debugging small chunks of code.
#define TEST_INTENSITY_4 (4)

#ifndef TEST_INTENSITY
// low-mid intensity by default
#define TEST_INTENSITY (2)
#endif

#define INTENSITY_1 (TEST_INTENSITY == TEST_INTENSITY_1)
#define INTENSITY_2 (TEST_INTENSITY == TEST_INTENSITY_2)
#define INTENSITY_3 (TEST_INTENSITY == TEST_INTENSITY_3)
#define INTENSITY_4 (TEST_INTENSITY == TEST_INTENSITY_4)