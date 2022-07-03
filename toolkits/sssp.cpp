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

typedef float Weight;

void compute(Graph<Weight> *graph, UINT root) {
    double exec_time = 0;
    exec_time -= get_time();
    graph->exe_data = 0;

    Weight *distance = graph->alloc_vertex_array<Weight>();
    VertexSubset *active_in = graph->alloc_vertex_subset();
    VertexSubset *active_out = graph->alloc_vertex_subset();
    active_in->clear();
    active_in->set_bit(root);
    graph->fill_vertex_array(distance, (Weight) 1e9);
    distance[root] = (Weight) 0;
    UINT active_vertices = 1;

    for (int i_i = 0; active_vertices > 0; i_i++) {
        if (graph->partition_idx == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
        }
        active_out->clear();
        active_vertices = graph->process_edges<UINT, Weight>(
                [&](UINT src) {
                    graph->emit(src, distance[src]);
                },
                [&](UINT src, Weight msg, AdjacentListType<Weight> outgoing_adj) {
                    UINT activated = 0;
                    for (EdgeType<Weight> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        UINT dst = ptr->dst;
                        Weight relax_dist = msg + ptr->edge_data;
                        if (relax_dist < distance[dst]) {
                            if (write_min(&distance[dst], relax_dist)) {
                                active_out->set_bit(dst);
                                activated += 1;
                            }
                        }
                    }
                    return activated;
                },
                [&](UINT dst, AdjacentListType<Weight> incoming_adj) {
                    Weight msg = 1e9;
                    for (EdgeType<Weight> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        UINT src = ptr->src;
                        // if (active_in->get_bit(src)) {
                        Weight relax_dist = distance[src] + ptr->edge_data;
                        if (relax_dist < msg) {
                            msg = relax_dist;
                        }
                        // }
                    }
                    if (msg < 1e9) graph->emit(dst, msg);
                },
                [&](UINT dst, Weight msg) {
                    if (msg < distance[dst]) {
                        write_min(&distance[dst], msg);
                        active_out->set_bit(dst);
                        return 1;
                    }
                    return 0;
                },
                active_in
        );
        std::swap(active_in, active_out);
    }

    exec_time += get_time();

    graph->gather_vertex_array(distance, 0);
    if (graph->partition_idx == 0) {
        UINT max_v_i = root;
        for (UINT v_i = 0; v_i < graph->vertices_num; v_i++) {
            if (distance[v_i] < 1e9 && distance[v_i] > distance[max_v_i]) {
                max_v_i = v_i;
            }
        }
        printf("distance[%u]=%f\n", max_v_i, distance[max_v_i]);
    }

    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->exe_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    if (graph->partition_idx == 0)
        printf("algo data: %lu MB, pre data: %lu MB\npre time: %lfs, exec time: %lf s\n", (graph->exe_data)>>20 ,(graph->prep_data)>>20, graph->prep_time, exec_time);

    graph->dealloc_vertex_array(distance);
    delete active_in;
    delete active_out;
}

int main(int argc, char **argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("sssp [file] [vertices_num] [root] [eta]\n");
        exit(-1);
    }

    Graph<Weight> *graph;
    graph = new Graph<Weight>(std::atoi(argv[4]));
    graph->load_directed(argv[1], std::atoi(argv[2]));
    UINT root = std::atoi(argv[3]);
    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->prep_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);

    graph->use_undirected=false;
    graph->use_root=true;

    compute(graph, root);
    for (int run = 0; run < 3; run++) {
        compute(graph, root);
    }

    delete graph;
    return 0;
}
