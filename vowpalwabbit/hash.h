/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#pragma once

#include <sys/types.h>  // defines size_t

// Platform-specific functions and macros
#if defined(_MSC_VER)                       // Microsoft Visual Studio
#   include <stdint.h>

#   include <stdlib.h>
#   define ROTL32(x,y)  _rotl(x,y)
#   define BIG_CONSTANT(x) (x)

#else                                       // Other compilers
#   include <stdint.h>   // defines uint32_t etc

inline uint32_t rotl32(uint32_t x, int8_t r)
{ return (x << r) | (x >> (32 - r));
}

#   define ROTL32(x,y)     rotl32(x,y)
#   define BIG_CONSTANT(x) (x##LLU)

#endif                                      // !defined(_MSC_VER)

namespace MURMUR_HASH_3
{

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

static inline uint32_t fmix(uint32_t h)
{ h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}
}

const uint32_t hash_base = 0;

uint64_t uniform_hash(const void *key, size_t length, uint64_t seed);

//-----------------------------------------------------------------------------
// compile time hash, thanks to Sam Hocevar
//   http://lolengine.net/blog/2011/12/20/cpp-constant-string-hash
#define __CTHASH1(s,i,x)   (x*65599u+(uint8_t)s[(i)<strlen(s)?strlen(s)-1-(i):strlen(s)])
#define __CTHASH4(s,i,x)   __CTHASH1(s,i,__CTHASH1(s,i+1,__CTHASH1(s,i+2,__CTHASH1(s,i+3,x))))
#define __CTHASH16(s,i,x)  __CTHASH4(s,i,__CTHASH4(s,i+4,__CTHASH4(s,i+8,__CTHASH4(s,i+12,x))))
#define __CTHASH64(s,i,x)  __CTHASH16(s,i,__CTHASH16(s,i+16,__CTHASH16(s,i+32,__CTHASH16(s,i+48,x))))
#define __CTHASH256(s,i,x) __CTHASH64(s,i,__CTHASH64(s,i+64,__CTHASH64(s,i+128,__CTHASH64(s,i+192,x))))
#define HASHSTR(s)         ((uint32_t)(__CTHASH256(s,0,0)^(__CTHASH256(s,0,0)>>16)))

