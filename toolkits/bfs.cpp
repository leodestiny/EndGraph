/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

void compute(Graph<Empty> *graph, UINT root) {
    double exec_time = 0;
    exec_time -= get_time();
    graph->exe_data = 0;

    UINT *parent = graph->alloc_vertex_array<UINT>();
    VertexSubset *visited = graph->alloc_vertex_subset();
    VertexSubset *active_in = graph->alloc_vertex_subset();
    VertexSubset *active_out = graph->alloc_vertex_subset();

    visited->clear();
    visited->set_bit(root);
    active_in->clear();
    active_in->set_bit(root);
    graph->fill_vertex_array(parent, graph->vertices_num);
    parent[root] = root;

    UINT active_vertices = 1;

    for (int i_i = 0; active_vertices > 0; i_i++) {
        if (graph->partition_idx == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
        }
        active_out->clear();
        active_vertices = graph->process_edges<UINT, UINT>(
                [&](UINT src) {
                    graph->emit(src, src);
                },
                [&](UINT src, UINT msg, AdjacentListType<Empty> outgoing_adj) {
                    UINT activated = 0;
                    for (EdgeType<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        UINT dst = ptr->dst;
                        if (parent[dst] == graph->vertices_num && cas(&parent[dst], graph->vertices_num, src)) {
                            active_out->set_bit(dst);
                            activated += 1;
                        }
                    }
                    return activated;
                },
                [&](UINT dst, AdjacentListType<Empty> incoming_adj) {
                    if (visited->get_bit(dst)) return;
                    for (EdgeType<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        UINT src = ptr->src;
                        if (active_in->get_bit(src)) {
                            graph->emit(dst, src);
                            break;
                        }
                    }
                },
                [&](UINT dst, UINT msg) {
                    if (cas(&parent[dst], graph->vertices_num, msg)) {
                        active_out->set_bit(dst);
                        return 1;
                    }
                    return 0;
                },
                active_in, visited
        );
        active_vertices = graph->process_vertices<UINT>(
                [&](UINT vtx) {
                    visited->set_bit(vtx);
                    return 1;
                },
                active_out
        );
        std::swap(active_in, active_out);
    }

    exec_time += get_time();

    graph->gather_vertex_array(parent, 0);
    if (graph->partition_idx == 0) {
        UINT found_vertices = 0;
        for (UINT v_i = 0; v_i < graph->vertices_num; v_i++) {
            if (parent[v_i] < graph->vertices_num) {
                found_vertices += 1;
            }
        }
        printf("found_vertices = %u\n", found_vertices);
    }
    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->exe_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    if (graph->partition_idx == 0)
        printf("algo data: %lu MB, pre data: %lu MB\npre time: %lf s, exec time: %lf\n", (graph->exe_data)>>20 ,(graph->prep_data)>>20, graph->prep_time, exec_time);


    graph->dealloc_vertex_array(parent);
    delete active_in;
    delete active_out;
    delete visited;
}

int main(int argc, char **argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("bfs [file] [vertices_num] [root] [eta]\n");
        exit(-1);
    }

    Graph<Empty> *graph;
    graph = new Graph<Empty>(std::atoi(argv[4]));
    UINT root = std::atoi(argv[3]);
    graph->load_directed(argv[1], std::atoi(argv[2]));
    graph->use_undirected=false;
    graph->use_root=true;
    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->prep_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);

    compute(graph, root);
    for (int run = 0; run < 3; run++) {
        compute(graph, root);
    }

    delete graph;
    return 0;
}
