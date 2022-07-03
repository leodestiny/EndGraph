# EndGraph

This project is an open source project for our paper (ICDCS'22: EndGraph: An Efficient Distributed Graph Preprocessing System).
In this paper, we propose a novel partition algorithm, named, CPBT to balance the preprocessing workload with lower bound of tim complexity. To construct distributed graph efficiently, we propose a two-level construction, including intra-machine construction and inter-machine construction.


This project extends state-of-the-art distributed graph processing system Gemini [OSDI 2016] and replaces its preprocessing(core/graph.hpp).
Hence, to compile and run EndGraph, you should follow the compile struction in the following readme of Gemini.

##Reference:
Tianfeng Liu, Dan Li

EndGraph: An Efficient Distributed Graph Preprocessing System (ICDCS 2022)

The following is coping readme fron original Gemini project.

# Gemini
A computation-centric distributed graph processing system.

## Quick Start
Gemini uses **MPI** for inter-process communication and **libnuma** for NUMA-aware memory allocation.
A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

Implementations of five graph analytics applications (PageRank, Connected Components, Single-Source Shortest Paths, Breadth-First Search, Betweenness Centrality) are inclulded in the *toolkits/* directory.

To build:
```
make
```

The input parameters of these applications are as follows:
```
./toolkits/pagerank [path] [vertices_num] [iterations]
./toolkits/cc [path] [vertices_num]
./toolkits/sssp [path] [vertices_num] [root]
./toolkits/bfs [path] [vertices_num] [root]
./toolkits/bc [path] [vertices_num] [root]
```

*[path]* gives the path of an input graph, i.e. a file stored on a *shared* file system, consisting of *|E|* \<source vertex id, destination vertex id, edge data\> tuples in binary.
*[vertices_num]* gives the number of vertices_num *|V|*. Vertex IDs are represented with 32-bit integers and edge data can be omitted for unweighted graphs (e.g. the above applications except SSSP).
Note: CC makes the input graph undirected by adding a reversed edge to the graph for each loaded one; SSSP uses *float* as the type of weights.

If Slurm is installed on the cluster, you may run jobs like this, e.g. 20 iterations of PageRank on the *twitter-2010* graph:
```
srun -N 8 ./toolkits/pagerank /path/to/twitter-2010.binedgelist 41652230 20
```

## Resources

Xiaowei Zhu, Wenguang Chen, Weimin Zheng, and Xiaosong Ma.
Gemini: A Computation-Centric Distributed Graph Processing System.
12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16).

