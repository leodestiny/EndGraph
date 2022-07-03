/*
   Copyright (c) 2019-2022 Tianfeng Liu, Tsinghua University
   Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef TYPE_HPP
#define TYPE_HPP

#include <stdint.h>

struct Empty {
};

typedef uint32_t UINT;
typedef uint64_t ULLONG;

template<typename EdgeData>
struct EdgeType {
    UINT src;
    UINT dst;
    EdgeData edge_data;
} __attribute__((packed));

template<>
struct EdgeType<Empty> {
    UINT src;
    union {
        UINT dst;
        Empty edge_data;
    };
} __attribute__((packed));

template<typename EdgeData>
struct AdjacentType {
    UINT neighbour;
    EdgeData edge_data;
} __attribute__((packed));

template<>
struct AdjacentType<Empty> {
    union {
        UINT neighbour;
        Empty edge_data;
    };
} __attribute__((packed));

struct CompressedAdjIndexType {
    ULLONG index;
    UINT vertex;
    CompressedAdjIndexType(ULLONG i, UINT v):index(i),vertex(v){}
    CompressedAdjIndexType(){}
} __attribute__((packed));

template<typename EdgeData>
struct AdjacentListType {
    EdgeType<EdgeData> *begin;
    EdgeType<EdgeData> *end;

    AdjacentListType() : begin(nullptr), end(nullptr) {}

    AdjacentListType(EdgeType<EdgeData> *begin, EdgeType<EdgeData> *end) : begin(begin), end(end) {}
};

#endif
