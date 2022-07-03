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

void compute(Graph<Empty> *graph) {
    double exec_time = 0;
    exec_time -= get_time();
    graph->exe_data = 0;

    UINT *label = graph->alloc_vertex_array<UINT>();
    VertexSubset *active_in = graph->alloc_vertex_subset();
    active_in->fill();
    VertexSubset *active_out = graph->alloc_vertex_subset();

    UINT active_vertices = graph->process_vertices<UINT>(
            [&](UINT vtx) {
                label[vtx] = vtx;
                return 1;
            },
            active_in
    );

    for (int i_i = 0; active_vertices > 0; i_i++) {
        if (graph->partition_idx == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
        }
        active_out->clear();
        active_vertices = graph->process_edges<UINT, UINT>(
                [&](UINT src) {
                    graph->emit(src, label[src]);
                },
                [&](UINT src, UINT msg, AdjacentListType<Empty> outgoing_adj) {
                    UINT activated = 0;
                    for (EdgeType<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        UINT dst = ptr->dst;
                        if (msg < label[dst]) {
                            write_min(&label[dst], msg);
                            active_out->set_bit(dst);
                            activated += 1;
                        }
                    }
                    return activated;
                },
                [&](UINT dst, AdjacentListType<Empty> incoming_adj) {
                    UINT msg = dst;
                    for (EdgeType<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        UINT src = ptr->src;
                        if (label[src] < msg) {
                            msg = label[src];
                        }
                    }
                    if (msg < dst) {
                        graph->emit(dst, msg);
                    }
                },
                [&](UINT dst, UINT msg) {
                    if (msg < label[dst]) {
                        write_min(&label[dst], msg);
                        active_out->set_bit(dst);
                        return 1u;
                    }
                    return 0u;
                },
                active_in
        );
        std::swap(active_in, active_out);
    }

    exec_time += get_time();

    graph->gather_vertex_array(label, 0);
    if (graph->partition_idx == 0) {
        UINT *count = graph->alloc_vertex_array<UINT>();
        graph->fill_vertex_array(count, 0u);
        for (UINT v_i = 0; v_i < graph->vertices_num; v_i++) {
            count[label[v_i]] += 1;
        }
        UINT components = 0;
        for (UINT v_i = 0; v_i < graph->vertices_num; v_i++) {
            if (count[v_i] > 0) {
                components += 1;
            }
        }
        printf("components = %u\n", components);
    }

    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->exe_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    if (graph->partition_idx == 0)
    printf("algo data: %lu MB, pre data: %lu MB\npre time: %lf s, exec time: %lf s \n", (graph->exe_data)>>20 ,(graph->prep_data)>>20, graph->prep_time, exec_time);

    graph->dealloc_vertex_array(label);
    delete active_in;
    delete active_out;
}

int main(int argc, char **argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 3) {
        printf("cc [file] [vertices_num] [null] [eta]\n");
        exit(-1);
    }

    Graph<Empty> *graph;
    graph = new Graph<Empty>(std::atoi(argv[4]));
    graph->load_undirected_from_directed(argv[1], std::atoi(argv[2]));
    graph->use_undirected=true;
    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->prep_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);

    compute(graph);
    for (int run = 0; run < 3; run++) {
        compute(graph);
    }

    delete graph;
    return 0;
}
