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

#include <math.h>

const double d = (double) 0.85;

void compute(Graph<Empty> *graph, int iterations) {
    double exec_time = 0;
    exec_time -= get_time();
    graph->exe_data = 0;

    double *curr = graph->alloc_vertex_array<double>();
    double *next = graph->alloc_vertex_array<double>();
    VertexSubset *active = graph->alloc_vertex_subset();
    active->fill();

    double delta = graph->process_vertices<double>(
            [&](UINT vtx) {
                curr[vtx] = (double) 1;
                if (graph->degree[vtx] > 0) {
                    curr[vtx] /= graph->degree[vtx];
                }
                return (double) 1;
            },
            active
    );
    delta /= graph->vertices_num;

    for (int i_i = 0; i_i < iterations; i_i++) {
        if (graph->partition_idx == 0) {
            printf("delta(%d)=%lf\n", i_i, delta);
        }
        graph->fill_vertex_array(next, (double) 0);
        graph->process_edges<int, double>(
                [&](UINT src) {
                    graph->emit(src, curr[src]);
                },
                [&](UINT src, double msg, AdjacentListType<Empty> outgoing_adj) {
                    for (EdgeType<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        UINT dst = ptr->dst;
                        write_add(&next[dst], msg);
                    }
                    return 0;
                },
                [&](UINT dst, AdjacentListType<Empty> incoming_adj) {
                    double sum = 0;
                    for (EdgeType<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        UINT src = ptr->src;
                        sum += curr[src];
                    }
                    graph->emit(dst, sum);
                },
                [&](UINT dst, double msg) {
                    write_add(&next[dst], msg);
                    return 0;
                },
                active
        );
        if (i_i == iterations - 1) {
            delta = graph->process_vertices<double>(
                    [&](UINT vtx) {
                        next[vtx] = 1 - d + d * next[vtx];
                        return 0;
                    },
                    active
            );
        } else {
            delta = graph->process_vertices<double>(
                    [&](UINT vtx) {
                        next[vtx] = 1 - d + d * next[vtx];
                        if (graph->degree[vtx] > 0) {
                            next[vtx] /= graph->degree[vtx];
                            return fabs(next[vtx] - curr[vtx]) * graph->degree[vtx];
                        }
                        return fabs(next[vtx] - curr[vtx]);
                    },
                    active
            );
        }
        delta /= graph->vertices_num;
        std::swap(curr, next);
    }

    exec_time += get_time();

    double pr_sum = graph->process_vertices<double>(
            [&](UINT vtx) {
                return curr[vtx];
            },
            active
    );
    if (graph->partition_idx == 0) {
        printf("pr_sum=%lf\n", pr_sum);
    }

    graph->gather_vertex_array(curr, 0);
    if (graph->partition_idx == 0) {
        UINT max_v_i = 0;
        for (UINT v_i = 0; v_i < graph->vertices_num; v_i++) {
            if (curr[v_i] > curr[max_v_i]) max_v_i = v_i;
        }
        printf("pr[%u]=%lf\n", max_v_i, curr[max_v_i]);
    }

    MPI_Datatype dt = get_mpi_data_type<ULLONG>();
    MPI_Allreduce(MPI_IN_PLACE, &graph->exe_data, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    if(graph->partition_idx == 0)
        printf("algo data: %lu MB, pre data: %lu MB\npre time: %lf s exec time: %lf s\n", (graph->exe_data)>>20 ,(graph->prep_data)>>20, graph->prep_time,exec_time);

    graph->dealloc_vertex_array(curr);
    graph->dealloc_vertex_array(next);
    delete active;
}

int main(int argc, char **argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("pagerank [file] [vertices_num] [iterations] [eta]\n");
        exit(-1);
    }

    Graph<Empty> *graph;
    graph = new Graph<Empty>(std::atoi(argv[4]));
    graph->load_directed(argv[1], std::atoi(argv[2]));
    MPI_Allreduce(MPI_IN_PLACE, &graph->prep_data, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    int iterations = std::atoi(argv[3]);
    graph->use_undirected=false;
    graph->use_iteration=true;
    graph->max_iterations=iterations;
   

    compute(graph, iterations);
    for (int run = 0; run < 3; run++) {
        compute(graph, iterations);
    }

    delete graph;
    return 0;
}
