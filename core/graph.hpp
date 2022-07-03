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

#ifndef GRAPH_HPP
#define GRAPH_HPP

#define PRINT_DEBUG_MESSAGES
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>
#include <errno.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"


enum ThreadStatus {
    WORKING,
    STEALING
};

enum MessageTag {
    ShuffleGraph,
    PassMessage,
    GatherVertexArray
};

struct ThreadState {
    UINT curr;
    UINT end;
    ThreadStatus status;
};

struct MessageBuffer {
    size_t capacity;
    int count; // the actual size (i.e. bytes) should be sizeof(element) * count
    char *data;

    MessageBuffer() {
        capacity = 0;
        count = 0;
        data = NULL;
    }

    void init(int socket_id) {
        capacity = 4096;
        count = 0;
        data = (char *) numa_alloc_onnode(capacity, socket_id);
    }

    void resize(size_t new_capacity) {
        if (new_capacity > capacity) {
            char *new_data = (char *) numa_realloc(data, capacity, new_capacity);
            assert(new_data != NULL);
            data = new_data;
            capacity = new_capacity;
        }
    }
};

template<typename MsgData>
struct MsgUnit {
    UINT vertex;
    MsgData msg_data;
} __attribute__((packed));

template<typename EdgeData = Empty>
class Graph {
    public:

        bool has_iterations;
        bool has_root;
        bool use_undirected;

        int max_iterations;

        // coff to calculate global partition
        size_t eta;

        // operating system and cluster parameters
        // global partition numbers on entire cluster
        int partitions_num;
        // partition_idx on this worker
        int partition_idx;
        // total threads number on one worker
        int threads_num;
        // total sockets number on one worker
        int sockets_num;
        // as variable name
        int threads_per_socket;
        int computation_threads_num;

        EdgeType<EdgeData> *edge_list;

        ULLONG exe_data;
        ULLONG prep_data;

        // number of vertices on entire graph data
        UINT vertices_num;
        // number of edges this worker read from file
        ULLONG read_edges_num;
        ULLONG edges_num;
        UINT pages_num;

        double prep_time = 0;
        // counting in degree and out degree of every vertices
        UINT *degree; // UINT [vertices_num]; numa-aware

        bool sparse;

        // record sub-graphs start and end range on every worker
        UINT *partition_offset; // UINT [partitions_num+1]
        // record every sub-graph start and end range on every sockets
        UINT *local_partition_offset; // UINT [sockets_num+1]
        UINT *page_offset;
        UINT *local_page_offset;

        // after graph splitting, the number of vertices this worker has been assigned
        UINT owned_vertices;

        EdgeType<EdgeData> *inter_incoming_edge_array;
        EdgeType<EdgeData> *inter_outgoing_edge_array;

        UINT inter_incoming_edge_array_size;
        UINT inter_outgoing_edge_array_size;


        ULLONG *outgoing_edges; // ULLONG [sockets_num]
        ULLONG *incoming_edges; // ULLONG [sockets_num]

        BitmapType **incoming_adj_bitmap;
        ULLONG **incoming_adj_index; // ULLONG [sockets_num] [vertices_num+1]; numa-aware
        EdgeType<EdgeData> **incoming_adj_list; // AdjacentType<EdgeData> [sockets_num] [vertices_num+1]; numa-aware
        BitmapType **outgoing_adj_bitmap;
        ULLONG **outgoing_adj_index; // ULLONG [sockets_num] [vertices_num+1]; numa-aware
        EdgeType<EdgeData> **outgoing_adj_list; // AdjacentType<EdgeData> [sockets_num] [vertices_num+1]; numa-aware

        UINT *compressed_incoming_adj_vertices;
        CompressedAdjIndexType **compressed_incoming_adj_index; // CompressedAdjIndexType [sockets_num] [...+1]; numa-aware
        UINT *compressed_outgoing_adj_vertices;
        CompressedAdjIndexType **compressed_outgoing_adj_index; // CompressedAdjIndexType [sockets_num] [...+1]; numa-aware

        // per thread state, it is numa-aware, will put onto proper sockets, using thread_idx
        ThreadState **thread_state; // ThreadState* [threads_num]; numa-aware
        ThreadState **tuned_chunks_dense; // ThreadState [partitions_num][threads_num];
        ThreadState **tuned_chunks_sparse; // ThreadState [partitions_num][threads_num];

        int  local_send_buffer_limit;

        // temporary thread local send buffer
        // one thread, one local_send_buffer
        // alloc on socket of each thread
        // thread will firstly write sending data to local_send_buffer
        // when local_send_buffer is full, it will flush into socket send buffer
        MessageBuffer **local_send_buffer; // MessageBuffer* [threads_num]; numa-aware

        // in computation phase, record which partition should send to
        int current_send_part_idx;

        double exec_time = 0;

        // two dimension
        // every buffer will only numa malloc on proper socket, so they are numa-aware
        // send_buffer[p_i][s_i] means sending message to socket s_i on partition_idx p_i
        // recv_buffer[p_i][s_i] means receiving message from socket s_i on partiton_idx p_i
        MessageBuffer ***send_buffer; // MessageBuffer* [partitions_num] [sockets_num]; numa-aware
        MessageBuffer ***recv_buffer; // MessageBuffer* [partitions_num] [sockets_num]; numa-aware

        Graph(int a) {
            // get current operating system config
            threads_num = numa_num_configured_cpus();
            sockets_num = numa_num_configured_nodes();
            threads_per_socket = threads_num / sockets_num;


            init();
            eta = a;
        }

        // using thread_idx to get which socket it in
        inline int get_socket_idx(int thread_idx) {
            return thread_idx / threads_per_socket;
        }



        //using thread_idx to get its offset on its socket
        inline int get_socket_offset(int thread_idx) {
            return thread_idx % threads_per_socket;
        }

        void init() {

            assert(numa_available() != -1);
            assert(sizeof(unsigned long) == 8); // assume unsigned long is 64-bit

            exe_data = 0;
            prep_data = 0;


            // initialize numa config
            char nodestring[sockets_num * 2 + 1];
            nodestring[0] = '0';
            for (int s_i = 1; s_i < sockets_num; s_i++) {
                nodestring[s_i * 2 - 1] = ',';
                nodestring[s_i * 2] = '0' + s_i;
            }
            struct bitmask *nodemask = numa_parse_nodestring(nodestring);
            numa_set_interleave_mask(nodemask);

            // initialize openmp config
            omp_set_dynamic(0);
            omp_set_num_threads(threads_num);

            thread_state = new ThreadState *[threads_num];
            local_send_buffer_limit = 16;
            local_send_buffer = new MessageBuffer *[threads_num];
            for (int t_i = 0; t_i < threads_num; t_i++) {
                thread_state[t_i] = (ThreadState *) numa_alloc_onnode(sizeof(ThreadState), get_socket_idx(t_i));
                local_send_buffer[t_i] = (MessageBuffer *) numa_alloc_onnode(sizeof(MessageBuffer), get_socket_idx(t_i));
                local_send_buffer[t_i]->init(get_socket_idx(t_i));
            }
#pragma omp parallel for
            for (int t_i = 0; t_i < threads_num; t_i++) {
                int s_i = get_socket_idx(t_i);
                assert(numa_run_on_node(s_i) == 0);
            }



            MPI_Comm_rank(MPI_COMM_WORLD, &partition_idx);
            MPI_Comm_size(MPI_COMM_WORLD, &partitions_num);
            send_buffer = new MessageBuffer **[partitions_num];
            recv_buffer = new MessageBuffer **[partitions_num];
            for (int i = 0; i < partitions_num; i++) {
                send_buffer[i] = new MessageBuffer *[sockets_num];
                recv_buffer[i] = new MessageBuffer *[sockets_num];
                for (int s_i = 0; s_i < sockets_num; s_i++) {
                    send_buffer[i][s_i] = (MessageBuffer *) numa_alloc_onnode(sizeof(MessageBuffer), s_i);
                    send_buffer[i][s_i]->init(s_i);
                    recv_buffer[i][s_i] = (MessageBuffer *) numa_alloc_onnode(sizeof(MessageBuffer), s_i);
                    recv_buffer[i][s_i]->init(s_i);
                }
            }

            computation_threads_num = threads_num;
            if(partitions_num != 1)
                computation_threads_num -= 1;


            MPI_Barrier(MPI_COMM_WORLD);
        }

        // fill a vertex array with a specific value
        template<typename T>
            void fill_vertex_array(T *array, T value) {
#pragma omp parallel for
                for (UINT v_i = partition_offset[partition_idx]; v_i < partition_offset[partition_idx + 1]; v_i++) {
                    array[v_i] = value;
                }
            }

        // allocate a numa-aware vertex array
        template<typename T>
            T *alloc_vertex_array() {
                char *array = (char *) mmap(NULL, sizeof(T) * vertices_num, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0);
                assert(array != NULL);
                for (int s_i = 0; s_i < sockets_num; s_i++) {
                    numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i],
                            sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
                }
                return (T *) array;
            }

        // deallocate a vertex array
        template<typename T>
            void dealloc_vertex_array(T *array) {
                numa_free(array, sizeof(T) * vertices_num);
            }

        // allocate a numa-oblivious vertex array
        template<typename T>
            T *alloc_interleaved_vertex_array() {
                T *array = (T *) numa_alloc_interleaved(sizeof(T) * vertices_num);
                assert(array != NULL);
                return array;
            }

        // dump a vertex array to path
        template<typename T>
            void dump_vertex_array(T *array, std::string path) {
                long file_length = sizeof(T) * vertices_num;
                if (!file_exists(path) || file_size(path) != file_length) {
                    if (partition_idx == 0) {
                        FILE *fout = fopen(path.c_str(), "wb");
                        char *buffer = new char[PAGESIZE];
                        for (long offset = 0; offset < file_length;) {
                            if (file_length - offset >= PAGESIZE) {
                                fwrite(buffer, 1, PAGESIZE, fout);
                                offset += PAGESIZE;
                            } else {
                                fwrite(buffer, 1, file_length - offset, fout);
                                offset += file_length - offset;
                            }
                        }
                        fclose(fout);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                int fd = open(path.c_str(), O_RDWR);
                assert(fd != -1);
                long offset = sizeof(T) * partition_offset[partition_idx];
                long end_offset = sizeof(T) * partition_offset[partition_idx + 1];
                void *data = (void *) array;
                assert(lseek(fd, offset, SEEK_SET) != -1);
                while (offset < end_offset) {
                    long bytes = write(fd, data + offset, end_offset - offset);
                    assert(bytes != -1);
                    offset += bytes;
                }
                assert(close(fd) == 0);
            }

        // restore a vertex array from path
        template<typename T>
            void restore_vertex_array(T *array, std::string path) {
                long file_length = sizeof(T) * vertices_num;
                if (!file_exists(path) || file_size(path) != file_length) {
                    assert(false);
                }
                int fd = open(path.c_str(), O_RDWR);
                assert(fd != -1);
                long offset = sizeof(T) * partition_offset[partition_idx];
                long end_offset = sizeof(T) * partition_offset[partition_idx + 1];
                void *data = (void *) array;
                assert(lseek(fd, offset, SEEK_SET) != -1);
                while (offset < end_offset) {
                    long bytes = read(fd, data + offset, end_offset - offset);
                    assert(bytes != -1);
                    offset += bytes;
                }
                assert(close(fd) == 0);
            }

        // gather a vertex array
        template<typename T>
            void gather_vertex_array(T *array, int root) {
                if (partition_idx != root) {
                    MPI_Send(array + partition_offset[partition_idx], sizeof(T) * owned_vertices, MPI_CHAR, root,
                            GatherVertexArray, MPI_COMM_WORLD);
                } else {
                    for (int i = 0; i < partitions_num; i++) {
                        if (i == partition_idx) continue;
                        MPI_Status recv_status;
                        MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]),
                                MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
                        int length;
                        MPI_Get_count(&recv_status, MPI_CHAR, &length);
                        assert((size_t)length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
                    }
                }
            }

        // allocate a vertex subset
        VertexSubset *alloc_vertex_subset() {
            return new VertexSubset(vertices_num);
        }

        int get_partition_id(UINT v_i) {
            for (int i = 0; i < partitions_num; i++) {
                if (v_i >= partition_offset[i] && v_i < partition_offset[i + 1]) {
                    return i;
                }
            }
            assert(false);
        }


        // the range is left closed, right open.
        int get_local_partition_id(UINT v_i) {
            for (int s_i = 0; s_i < sockets_num; s_i++) {
                if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i + 1]) {
                    return s_i;
                }
            }
            return 0;
        }

#define MAX_RADIX 8

        // a type that must hold MAX_RADIX bits
        typedef unsigned char bIndexT;

        // this function will do serial counting sort
        // n elements start from pointer in will be sorted
        // buckets have m elements
        // getKey is a function which extract sorting key from input
        // Tmp array will only record temporary info from getKey
        // counts array will record the number of each key, it is useful in distributed radix sort
        // offsets array is temporary array and record prefix sum of counting
        // zero will set starting value, maybe always 0?
        template<class E, class F, class bint>
            void serial_counting_sort(E *in, E *out, bIndexT *tmp,
                    bint *counts, bint *offsets,
                    UINT length, UINT bucket_num, F getKey) {

                for (long i = 0; i < bucket_num; i++)
                    counts[i] = 0;
                for (long j = 0; j < length; j++) {
                    bint k = tmp[j] = getKey(in[j]);
                    counts[k]++;
                }
                bint s = 0;
                for (long i = 0; i < bucket_num; i++) {
                    offsets[i] = s;
                    s += counts[i];
                }
                for (long j = 0; j < length; j++) {
                    out[offsets[tmp[j]]] = in[j];
                    offsets[tmp[j]]++;
                }
            }


        // this function will do radix sort, result is also in din array
        // input data is din array
        // the length of input data is length
        // getKey is a functional, should take a variable with type E and return a UINT type variable
        template<class E, class F>
            UINT* load_balance_radix_sort(E *din, UINT length, UINT bits, F getKey) {
                /*
#ifdef PRINT_DEBUG_MESSAGES
                if (partition_idx==0) {
                    printf("start radix sort.\n");
                }
#endif
*/

                // calculate sorted bits per round and rounds number
                UINT rounds = 1 + (bits - 1) / MAX_RADIX;
                // rbits will record how many bits sort in one round
                UINT rbits = 1 + (bits - 1) / rounds;
                // bit_offset will record the starting position in bits
                UINT bit_offset = 0;


                // calculate data length each thread should process
                // careful deal with length % threads_num != 0
                UINT t_length = (length + threads_num - 1) / threads_num;

                // allocate temporary data;
                // every thread should have a count
                // we need, out, tmp, counts, offsets
                E **t_out = new E *[threads_num];
                bIndexT **t_tmp = new bIndexT *[threads_num];
                UINT **t_counts = new UINT *[threads_num];
                UINT **t_offsets = new UINT *[threads_num];

                UINT * g_offsets;

                for (int t_i = 0; t_i < threads_num; ++t_i) {
                    int s_i = get_socket_idx(t_i);
                    t_out[t_i] = (E *) numa_alloc_onnode(sizeof(E) * t_length, s_i);
                    t_tmp[t_i] = (bIndexT *) numa_alloc_onnode(sizeof(bIndexT) * t_length, s_i);
                    t_counts[t_i] = (UINT *) numa_alloc_onnode(sizeof(UINT) * (1 << rbits), s_i);
                    t_offsets[t_i] = (UINT *) numa_alloc_onnode(sizeof(UINT) * (1 << rbits), s_i);
                }

                // iterate over all bits
                for (UINT round_i = 0; round_i < rounds; ++round_i, bit_offset += rbits) {

                    // deal with corner case
                    // in last round, if we don't have enough bits equal to rbits
                    // we only sort residual bits
                    if (bit_offset + rbits > bits)
                        rbits = bits - bit_offset;

                    int buckets_num = 1 << rbits;

                    // dispatch jobs between threads
#pragma omp parallel for
                    for (int t_i = 0; t_i < threads_num; ++t_i) {

                        // noted, the range of each job is [begin,end)
                        UINT begin = t_i * t_length;
                        UINT end = (t_i + 1) * t_length;

                        // deal with corner case
                        if (end > length)
                            end = length;

                        serial_counting_sort(din+begin, t_out[t_i], t_tmp[t_i], t_counts[t_i], t_offsets[t_i],end-begin, buckets_num,
                                [=](E e) {return ((1 << rbits) - 1) & (getKey(e) >> bit_offset);} );



                        // refresh offsets array
                        // this will helpful in next global info transmition
                        for(int b_i = buckets_num -1 ; b_i > 0; b_i--)
                            t_offsets[t_i][b_i] = t_offsets[t_i][b_i-1];
                        t_offsets[t_i][0] = 0;

                    }
                    // after this for loop, every thread has sorted their local data and record bucket number in t_counts;

                    // gather thread local counts info into global counts info
                    UINT * g_counts = new UINT [buckets_num];
                    std::memset(g_counts,0,sizeof(UINT)*buckets_num);

#pragma omp parallel for
                    for(int b_i = 0; b_i < buckets_num; ++b_i){
                        for(int t_i = 0; t_i < threads_num; ++t_i)
                            g_counts[b_i] += t_counts[t_i][b_i];
                    }

                    // calculate global counts prefix sum
                    g_offsets = new UINT [buckets_num+1];
                    g_offsets[0] = 0;
                    for(int b_i = 1; b_i <= buckets_num; ++b_i)
                        g_offsets[b_i] = g_offsets[b_i-1]+g_counts[b_i-1];

                    // move data from thread local t_out to global din array
#pragma omp parallel for
                    for(int b_i = 0; b_i < buckets_num; ++b_i){
                        int b_j = g_offsets[b_i];
                        for(int t_i = 0; t_i < threads_num; ++t_i){
                            int t_j = t_offsets[t_i][b_i];
                            for(UINT k = 0; k < t_counts[t_i][b_i]; ++k){
                                din[b_j] = t_out[t_i][t_j];
                                b_j++;
                                t_j++;
                            }
                        }
                    }

                }


                // free all temporary array
                for (int t_i = 0; t_i < threads_num; ++t_i) {
                    numa_free(t_out[t_i],sizeof(E) * (t_length));
                    numa_free(t_tmp[t_i],sizeof(bIndexT) * (t_length));
                    numa_free(t_counts[t_i],sizeof(UINT) * (1 << rbits));
                    numa_free(t_offsets[t_i],sizeof(UINT) * (1 << rbits));
                }


                return g_offsets;
            }

        bool judge(ULLONG s, ULLONG* prefix, UINT n, UINT k){
            ULLONG sum = 0;

            for(UINT i = 0; i < k ; ++i){
                sum = *(std::upper_bound(prefix, prefix + n + 1, sum + s) - 1);
            }

            return sum==prefix[n];
        }


        UINT * optimal_offset(ULLONG* a, UINT n, UINT k){
            ULLONG* prefix = new ULLONG[n+1];
            prefix[0] = 0;
            for(UINT i = 0; i < n; ++i)
                prefix[i + 1] = prefix[i] + a[i];

            ULLONG lb = prefix[0];
            ULLONG ub = prefix[n];

            while(lb + 1 < ub){
                ULLONG mid = lb + ((ub - lb) >> 1);
                if(judge(mid, prefix, n, k))
                    ub = mid;
                else
                    lb = mid;
            }

            UINT * offset = new UINT[k+1];
            offset[0] = 0;
            offset[k] = n;
            ULLONG sum= 0;

            for(UINT i = 0; i < k - 1; ++i){
                ULLONG * ptr = std::upper_bound(prefix, prefix + n + 1, sum + ub) - 1;
                offset[i+1] = ptr - prefix;
                sum = * ptr;
            }

            return offset;
        }

        // load a directed graph and make it undirected
        void load_undirected_from_directed(std::string filepath, UINT vertices) {

#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("start preprocessing. loading from directed graph file and make it undirected\n");
            }
#endif


            preload_graph_file(filepath,vertices,true);

            prep_time = 0;
            prep_time -= MPI_Wtime();
            split_graph();

            prep_time += MPI_Wtime();

        }

        // transpose the graph
        void transpose() {
            std::swap(outgoing_edges, incoming_edges);
            std::swap(outgoing_adj_index, incoming_adj_index);
            std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
            std::swap(outgoing_adj_list, incoming_adj_list);
            std::swap(tuned_chunks_dense, tuned_chunks_sparse);
            std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
            std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
        }


        // in this function, we will calculate out degree of every vertex
        // using this out degree, we can assign ownership of each vertex
        void assign_vertices() {
#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("start assign_vertices.\n");
            }
#endif


            // prepare mpi type and out degree array
            degree = alloc_interleaved_vertex_array<UINT>();


#pragma omp parallel num_threads(threads_num)
            {
                int t_i = omp_get_thread_num();
                numa_run_on_node(get_socket_idx(t_i));
                UINT range = (vertices_num + threads_num - 1) / threads_num;
                UINT begin = t_i * range;
                UINT end = std::min<UINT>(vertices_num, (t_i + 1) * range);

                for(UINT i = begin; i < end; ++i)
                    degree[i] = 0;
            }



            //double t = -get_time();
#pragma omp parallel num_threads(threads_num)
            {
                int t_i = omp_get_thread_num();
                numa_run_on_node(get_socket_idx(t_i));
                UINT range = (read_edges_num + threads_num - 1) / threads_num;
                UINT begin = t_i * range;
                UINT end = std::min<UINT>(read_edges_num, (t_i + 1) * range);

                if(sparse){
                    for(UINT i = begin; i < end; ++i){
                        __sync_fetch_and_add(&degree[edge_list[i].dst],1);
                    }
                }
                else{
                    for(UINT i = begin; i < end; ++i){
                        __sync_fetch_and_add(&degree[edge_list[i].src],1);
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // allreduce between all workers
            double t = - get_time();
            MPI_Datatype vid_t = get_mpi_data_type<UINT>();
            MPI_Allreduce(MPI_IN_PLACE, degree, vertices_num, vid_t, MPI_SUM, MPI_COMM_WORLD);
            t += get_time();
            printf("part  %d allreduce take %lf s\n",partition_idx,t);
            prep_data += vertices_num * sizeof(UINT);

            pages_num = (vertices_num + PAGESIZE - 1) / PAGESIZE;
            ULLONG * page_amount = (ULLONG*)numa_alloc_interleaved(sizeof(ULLONG)*pages_num);

#pragma omp parallel num_threads(threads_num)
            {
                int t_i = omp_get_thread_num();
                UINT range = (pages_num + threads_num - 1) / threads_num;
                UINT begin = t_i * range;
                UINT end = std::min<UINT>(pages_num, (t_i + 1) * range);

                for(UINT i = begin; i < end; ++i){
                    page_amount[i] = 0;
                    UINT v_begin = i * PAGESIZE;
                    UINT v_end = std::min<UINT>(vertices_num, (i+1) * PAGESIZE);
                    for(UINT v_i = v_begin; v_i < v_end; ++v_i){
                        if(dense){
                            page_amount[i] += (1 + (partitions_num - 1) / 16) * degree[v_i] + eta * (partitions_num - 1 + max_iterations * degree[v_i]);
                        }                       
                        else{
                            page_amount[i] += (1 + (partitions_num - 1) / 16) * degree[v_i] + eta * (partitions_num - 1 + degree[v_i]);
                        }
                    }
                }
            }

            page_offset = optimal_offset(page_amount, pages_num, partitions_num);
            partition_offset = new UINT[partitions_num + 1];
            partition_offset[0] = 0;
            partition_offset[partitions_num] = vertices_num;

            for(int i = 0; i < partitions_num - 1; ++i)
                partition_offset[i + 1] = page_offset[i + 1] * PAGESIZE;

            ULLONG num = 0;
            for(UINT i = partition_offset[partition_idx]; i < partition_offset[partition_idx+1]; ++i)
                num += degree[i];

            printf("part %d with %u vertices and %lu edges\n", partition_idx, partition_offset[partition_idx + 1] - partition_offset[partition_idx], num);



            // check consistency of partition boundaries
            // record owned vertices
            assert(partition_offset[partitions_num] == vertices_num);
            owned_vertices = partition_offset[partition_idx + 1] - partition_offset[partition_idx];
            UINT *global_partition_offset = new UINT[partitions_num + 1];
            MPI_Allreduce(partition_offset, global_partition_offset, partitions_num + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
            for (int i = 0; i <= partitions_num; i++) {
                assert(partition_offset[i] == global_partition_offset[i]);
            }
            MPI_Allreduce(partition_offset, global_partition_offset, partitions_num + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
            for (int i = 0; i <= partitions_num; i++) {
                assert(partition_offset[i] == global_partition_offset[i]);
            }
            delete[] global_partition_offset;

            local_page_offset = optimal_offset(page_amount + page_offset[partition_idx], page_offset[partition_idx + 1] - page_offset[partition_idx], sockets_num);

            local_partition_offset = new UINT[sockets_num + 1];
            local_partition_offset[0] = partition_offset[partition_idx];
            local_partition_offset[sockets_num] = partition_offset[partition_idx + 1];
            for(int i = 0; i < sockets_num - 1; ++i)
                local_partition_offset[i + 1] = local_page_offset[i + 1] * PAGESIZE + partition_offset[partition_idx];

            /*
            for(int s_i = 0; s_i < sockets_num; ++ s_i){
                num = 0;
                for(UINT i = local_partition_offset[s_i]; i < local_partition_offset[s_i+1]; ++i)
                    num += degree[i];
                printf("part %d socket %d with %u vertices and %lu edges\n", partition_idx, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], num);
                //printf("part %d socket %d from %u to %u \n", partition_idx, s_i, local_partition_offset[s_i], local_partition_offset[s_i+1]);
            }
            */

        }


        // shuffle reading edge list, and transmit to corresponding owned worker.
        // shuffle results will record in edge_array, array size will record in edge_array_size.
        // getKey is a lambda function, it will determine the ownership of each edge.
        template<class T>
            void shuffle_edge_list(EdgeType<EdgeData> *&edge_array, UINT &edge_array_size,
                    T getKey) {

#ifdef PRINT_DEBUG_MESSAGES
                if (partition_idx==0) {
                    printf("shuffle edge list to each workers.\n");
                }
#endif
                // deal with corner case, only one node
                // so, we don't need to do sorting and communication
                if(partitions_num == 1){
                    edge_array_size = read_edges_num;
                    edge_array = edge_list;
                    return;
                }

                load_balance_radix_sort(edge_list, read_edges_num, counting_bits(vertices_num), getKey);


                // each thread counting bucket of local range.
                UINT** threads_counting_bucket = new UINT* [threads_num];
#pragma omp parallel num_threads(threads_num)
                {
                    int t_i = omp_get_thread_num();
                    int s_i = get_socket_idx(t_i);
                    numa_run_on_node(s_i);
                    UINT range = (read_edges_num + threads_num - 1) / threads_num;
                    UINT begin = t_i * range;
                    UINT end = std::min<UINT>(read_edges_num, (t_i + 1) * range);

                    threads_counting_bucket[t_i] = new UINT[partitions_num];
                    UINT * bucket = threads_counting_bucket[t_i];
                    for(int i = 0; i < partitions_num; ++i)
                        bucket[i] = 0;

                    int cur_part = get_partition_id(getKey(edge_list[begin]));

                    for(UINT e_i = begin; e_i < end; ++e_i){
                        UINT key = getKey(edge_list[e_i]);
                        while(key >= partition_offset[cur_part+1])
                            cur_part++;
                        bucket[cur_part]++;
                    }
                }

                // according to partition_offset, split edge_list and transmit to corresponding worker
                // counting_bucket[i][j] means worker i have number of counting_bucket[i][j] edges sending to worker j;
                UINT * counting_bucket = new UINT[partitions_num * partitions_num];
                memset(counting_bucket,0,sizeof(UINT)*partitions_num*partitions_num);

                // gather information into worker level
                for(int i = 0; i < partitions_num; ++i)
                    for(int j = 0; j < threads_num; ++j)
                        counting_bucket[partition_idx * partitions_num + i] += threads_counting_bucket[j][i];

                MPI_Datatype vid_t = get_mpi_data_type<UINT>();
                // allreduce counting_bucket across all workers, after this operation, every worker knows all buckets information
                MPI_Allreduce(MPI_IN_PLACE, counting_bucket, partitions_num * partitions_num, vid_t, MPI_SUM, MPI_COMM_WORLD);



                // calculate send_counting prefix sum
                UINT *send_counting = new UINT[partitions_num];
                send_counting[0] = 0;
                for (int i = 1; i < partitions_num; ++i)
                    send_counting[i] = send_counting[i - 1] + counting_bucket[partition_idx*partitions_num + i-1];


                // compute receive starting position
                // because we should provide starting position, receive_counting[0] = 0;
                UINT *receive_counting = new UINT[partitions_num];
                receive_counting[0] = 0;
                for (int i = 1; i < partitions_num; ++i)
                    receive_counting[i] = receive_counting[i - 1] + counting_bucket[(i - 1)*partitions_num + partition_idx];

                // calculate owned_edges and receive_list_buffer
                edge_array_size = receive_counting[partitions_num - 1] + counting_bucket[(partitions_num - 1)*partitions_num + partition_idx];
                edge_array = (EdgeType<EdgeData>*)numa_alloc_interleaved(edge_array_size * sizeof(EdgeType<EdgeData>));

#pragma omp parallel num_threads(threads_num)
                {
                    int t_i = omp_get_thread_num();
                    int t_j = get_socket_offset(t_i);
                    int s_i = get_socket_idx(t_i);
                    numa_run_on_node(s_i);
                    int local_count = counting_bucket[partition_idx * partitions_num + partition_idx];
                    int range = (local_count + threads_per_socket - 1) / threads_per_socket;
                    UINT begin = t_j * range;
                    UINT end = std::min<UINT>(local_count, (t_j + 1) * range);
                    int dst = receive_counting[partition_idx];
                    int src = send_counting[partition_idx];

                    for(UINT i = begin; i < end; ++i)
                        edge_array[dst+i] = edge_list[src+i];
                }


                MPI_Request* requests_array = new MPI_Request[partitions_num - 1];
                MPI_Status* status_array = new MPI_Status[partitions_num - 1];

                // MPI_send can only handle max data size is INT_MAX
                ULLONG max_bytes = INT_MAX / sizeof(EdgeType<EdgeData>) * sizeof(EdgeType<EdgeData>);

                // send edges to other workers
                for (int i = 0; i < partitions_num; i++) {
                    if (partition_idx == i)
                        continue;

                    ULLONG bytes_to_send = sizeof(EdgeType<EdgeData>) * counting_bucket[partition_idx*partitions_num + i];
                    prep_data += bytes_to_send;
                    while(bytes_to_send > 0){
                        if(bytes_to_send >= max_bytes){
                            MPI_Isend(edge_list + send_counting[i],max_bytes ,MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD,requests_array);
                            send_counting[i]+= max_bytes/sizeof(EdgeType<EdgeData>);
                            bytes_to_send -= max_bytes;
                        }
                        else{
                            MPI_Isend(edge_list + send_counting[i],bytes_to_send ,MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD,requests_array);
                            bytes_to_send = 0;
                        }
                        MPI_Request_free(requests_array);
                    }
                }

                // send finished flag
                int cnt = 0;
                for (int i = 0; i < partitions_num; i++) {
                    if (partition_idx == i)
                        continue;
                    char c = 0;
                    MPI_Isend(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD,requests_array+cnt);
                    cnt++;
                }

                // count finished worker
                int finished_count = 0;
                // temp probe variables
                MPI_Status recv_status;

                // only need receive from other workers
                while (finished_count < partitions_num - 1) {
                    MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int i = recv_status.MPI_SOURCE;
                    assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions_num);
                    int recv_bytes;
                    MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);

                    // if receive only one bytes, it is a finished flag
                    if (recv_bytes == 1) {
                        finished_count += 1;
                        char c;
                        MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        continue;
                    }

                    // receive bytes should divided by sizeof(EdgeType<EdgeData>)
                    assert(recv_bytes % sizeof(EdgeType<EdgeData>) == 0);

                    MPI_Recv(edge_array + receive_counting[i], recv_bytes,
                            MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    receive_counting[i] += recv_bytes / sizeof(EdgeType<EdgeData>);
                }

                MPI_Waitall(partitions_num - 1, requests_array, status_array);
                for(int i = 0; i < cnt; ++i)
                    MPI_Request_free(requests_array+i);
                numa_free(edge_list, sizeof(EdgeType<EdgeData>) * read_edges_num);
                MPI_Barrier(MPI_COMM_WORLD);


                delete[] receive_counting;
                delete[] counting_bucket;
                delete[] send_counting;

            }


        // split entire graph
        // this function take three steps:
        // 1. assign ownership of each vertex according to out degree array
        // 2. shuffle edge list according to its src vertex, prepare for construct incoming data structures
        // 3. shuffle edge list according to its dst vertex, prepare for construct outgoing data structures
        void split_graph() {

            assign_vertices();
            double shuffle_time = 0;

            if(sparse){
                shuffle_time -= MPI_Wtime();
                shuffle_edge_list(inter_outgoing_edge_array, inter_outgoing_edge_array_size, [](EdgeType<EdgeData> e) {
                    return e.dst;
                });
                shuffle_time += MPI_Wtime();
                if (partition_idx==0) {
                    printf("shuffling edge takes %lf s", shuffle_time);
                }

                prepare_push_mode();
            }else{
                shuffle_time -= MPI_Wtime();
                shuffle_edge_list(inter_incoming_edge_array, inter_incoming_edge_array_size, [](EdgeType<EdgeData> e) {
                    return e.src;
                });
                shuffle_time += MPI_Wtime();
                if (partition_idx==0) {
                    printf("shuffling edge takes %lf s", shuffle_time);
                }
                prepare_pull_mode();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        void  preload_graph_file(std::string filepath, UINT vertices_num, bool symmetric) {

#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("preload graph file.\n");
            }
#endif
            // record total vertices number
            this->vertices_num = vertices_num;


            // read file bytes
            // determine which part of edge file, this worker should process
            ULLONG total_bytes = file_size(filepath.c_str());
            this->edges_num = total_bytes / sizeof(EdgeType<EdgeData>);
            read_edges_num = edges_num / partitions_num;
            if(partition_idx == partitions_num - 1)
                read_edges_num += edges_num % partitions_num;
            ULLONG bytes_to_read = sizeof(EdgeType<EdgeData>)* read_edges_num;
            ULLONG read_offset = sizeof(EdgeType<EdgeData>)*(edges_num / partitions_num * partition_idx);
            int fin = open(filepath.c_str(), O_RDONLY);
            lseek(fin,read_offset,SEEK_SET);


            // malloc edge list array, and copy from binary file
            // if symmetric flag is set, we should double array size.
            if (symmetric)
                edge_list = (EdgeType<EdgeData> *) numa_alloc_interleaved(2 * read_edges_num * sizeof(EdgeType<EdgeData>));
            else
                edge_list = (EdgeType<EdgeData> *) numa_alloc_interleaved(read_edges_num * sizeof(EdgeType<EdgeData>));

            ULLONG read_bytes = 0;
            read_edges_num = 0;
            while (read_bytes < bytes_to_read) {
                long curr_read_bytes;
                if (bytes_to_read - read_bytes > sizeof(EdgeType<EdgeData>) * CHUNKSIZE) {
                    curr_read_bytes = read(fin, edge_list+read_edges_num, sizeof(EdgeType<EdgeData>)* CHUNKSIZE);
                } else {
                    curr_read_bytes = read(fin, edge_list+read_edges_num, bytes_to_read - read_bytes);
                }
                read_edges_num += curr_read_bytes / sizeof(EdgeType<EdgeData>);
                read_bytes += curr_read_bytes;
            }


            // if graph is undirected,
            if (symmetric) {
#pragma omp parallel num_threads(threads_num)
                {
                    int t_i = omp_get_thread_num();
                    UINT range = (read_edges_num + threads_num - 1) / threads_num;
                    UINT begin = t_i * range;
                    UINT end = std::min<UINT>(read_edges_num, (t_i + 1) * range);

                    for(UINT i = begin; i < end; ++i){
                        UINT pos = i + read_edges_num;
                        edge_list[pos].dst = edge_list[i].src;
                        edge_list[pos].src = edge_list[i].dst;
                    }
                    if(!std::is_same<EdgeData, Empty>::value){
                        for(UINT i = begin; i < end; ++i){
                            edge_list[i + read_edges_num].edge_data = edge_list[i].edge_data;
                        }
                    }
                }
                read_edges_num *= 2;
                edges_num *= 2;
            }
            MPI_Barrier(MPI_COMM_WORLD);


            // close file
            close(fin);
        }

        // this function will return number of bits of parameter n
        int counting_bits(int n){
            int i = 0;
            for(;n > 0;n >>= 1)
                i++;
            return i;
        }

        // to run in pull mode, we need prepare all incoming data structures
        // we use receive_list_buffer as intermediate data structure
        // firstly, we sort receive_list_buffer according to dst vertex of each edge
        // secondly, we construct all incoming data structures from sorted receive_list_buffer
        void prepare_pull_mode() {

#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("prepare pull mode data structure.\n");
            }
#endif
            double sorting_time = 0;
            sorting_time -= MPI_Wtime();
            load_balance_radix_sort(inter_incoming_edge_array,inter_incoming_edge_array_size, counting_bits(vertices_num), [](EdgeType<EdgeData> e){ return e.dst;});
            sorting_time += MPI_Wtime();
            if (partition_idx==0) {
                printf("sorting edge takes %lf s", sorting_time);
            }

            UINT * g_offsets = load_balance_radix_sort(inter_incoming_edge_array,inter_incoming_edge_array_size,counting_bits(sockets_num),
                    [&](EdgeType<EdgeData> e){ return get_local_partition_id(e.src);});

            incoming_edges = new ULLONG[sockets_num];
            for(int i = 0; i < sockets_num; ++i)
                incoming_edges[i] = g_offsets[i+1] - g_offsets[i];


            incoming_adj_list = new EdgeType<EdgeData> *[sockets_num];
            for(int s_i = 0; s_i < sockets_num; ++s_i)
                incoming_adj_list[s_i] = (EdgeType<EdgeData> *) numa_alloc_onnode(sizeof(EdgeType<EdgeData>) * incoming_edges[s_i], s_i);


#pragma omp parallel num_threads(threads_num)
            {
                int t_i = omp_get_thread_num();
                int s_i = get_socket_idx(t_i);
                int t_j = get_socket_offset(t_i);
                int range = (incoming_edges[s_i] + threads_per_socket - 1) / threads_per_socket;
                UINT begin = t_j * range;
                UINT end = std::min<uint>(incoming_edges[s_i], (t_j + 1) * range);
                EdgeType<EdgeData>* adj_list = incoming_adj_list[s_i];
                UINT off = g_offsets[s_i];

                for(uint i = begin; i < end; ++i){
                    adj_list[i] = inter_incoming_edge_array[off+i];
                }
            }

            // assert in each socket, incoming edges has been sorted by dst;

            compressed_incoming_adj_vertices = new UINT[sockets_num];
            compressed_incoming_adj_index = new CompressedAdjIndexType *[sockets_num];

#pragma omp parallel num_threads(sockets_num)
            {
                int s_i = omp_get_thread_num();
                numa_run_on_node(s_i);
                ULLONG local_edges_num = incoming_edges[s_i];
                EdgeType<EdgeData>* adj_list = incoming_adj_list[s_i];

                UINT pre = adj_list[0].dst;
                UINT adj_vertices = 1;
                for(ULLONG i = 0; i < local_edges_num; ++i){
                    UINT dst = adj_list[i].dst;
                    if(pre != dst){
                        adj_vertices++;
                        pre = dst;
                    }
                }

                compressed_incoming_adj_vertices[s_i] = adj_vertices;
                compressed_incoming_adj_index[s_i] = (CompressedAdjIndexType *) numa_alloc_onnode(sizeof(CompressedAdjIndexType) * (adj_vertices + 1), s_i);
                CompressedAdjIndexType * adj_index = compressed_incoming_adj_index[s_i];

                // deal with start point
                pre = adj_list[0].dst;
                adj_index[0].index = 0;
                adj_index[0].vertex = pre;
                adj_vertices = 1;

                // loop entire array
                for(ULLONG i = 1; i < local_edges_num; ++i){
                    UINT dst = adj_list[i].dst;
                    if(pre != dst){
                        adj_index[adj_vertices].index = i;
                        adj_index[adj_vertices].vertex = dst;
                        adj_vertices++;
                        pre = dst;
                    }
                }

                // add boundary value
                adj_index[adj_vertices].index = local_edges_num;
                adj_index[adj_vertices].vertex = vertices_num;
            }


            tuned_chunks_dense = new ThreadState *[partitions_num];
            for(int i = 0; i < partitions_num; ++i)
                tuned_chunks_dense[i] = new ThreadState[threads_num];

#pragma omp parallel num_threads(partitions_num*sockets_num)
            {
                int t_id = omp_get_thread_num();
                int i = t_id / sockets_num;
                int s_i = t_id % sockets_num;

                UINT p_v_i = 0;
                UINT last_p_v_i, end_p_v_i;
                while(p_v_i < compressed_incoming_adj_vertices[s_i]){
                    if(compressed_incoming_adj_index[s_i][p_v_i].vertex >= partition_offset[i])
                        break;
                    p_v_i++;
                }
                last_p_v_i = p_v_i;
                while(p_v_i < compressed_incoming_adj_vertices[s_i]){
                    if(compressed_incoming_adj_index[s_i][p_v_i].vertex >= partition_offset[i+1])
                        break;
                    p_v_i++;
                }
                end_p_v_i = p_v_i;

                UINT length = end_p_v_i - last_p_v_i;
                ULLONG * a = new ULLONG[length];
                UINT cnt = 0;
                for (UINT p_v_i = last_p_v_i; p_v_i < end_p_v_i; cnt++, p_v_i++) {
                    ULLONG n = compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
                    a[cnt] += eta; 
                }

                // deal with corner case
                // because in computation phase, we use another thread to communicate
                // we ping the communication thread in the last socket
                // so, in the last socket, the number of computation thread should be decreased one
                int worker_num = threads_per_socket;
                UINT *offset = optimal_offset(a, length, worker_num);

                for(int s_j = 0; s_j < worker_num; ++s_j){
                    UINT t_i = s_i * threads_per_socket + s_j;
                    tuned_chunks_dense[i][t_i].status = WORKING;
                    tuned_chunks_dense[i][t_i].curr = last_p_v_i + offset[s_j];
                    tuned_chunks_dense[i][t_i].end = last_p_v_i + offset[s_j+1];
                }
            }

            numa_free(inter_incoming_edge_array, sizeof(EdgeType<EdgeData>) * inter_incoming_edge_array_size);
            inter_incoming_edge_array_size = 0;

        }

        // to run in push/sparse mode, we need prepare all outgoing data structures
        // we use receive_list_buffer as intermediate data structure
        // firstly, we sort receive_list_buffer according to src vertex of each edge
        // secondly, we construct all outgoing data structures from sorted receive_list_buffer
        void prepare_push_mode() {

#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("prepare push mode data structure.\n");
            }
#endif

            // firstly, we sort receive_list_buffer
            // first key is the socket id that the dst vertex of edge belongs to
            // second key is the src vertex of edge
            // in radix sort, we sort data array from lower bit to high bit, so, we do two rounds in reverse order
            double sorting_time = 0;
            sorting_time -= MPI_Wtime();
            load_balance_radix_sort(inter_outgoing_edge_array,inter_outgoing_edge_array_size, counting_bits(vertices_num),
                    [](EdgeType<EdgeData> e){ return e.src;});
            sorting_time += MPI_Wtime();
            if (partition_idx==0) {
                printf("sorting edge takes %lf s", sorting_time);
            }
            UINT * g_offsets = load_balance_radix_sort(inter_outgoing_edge_array,inter_outgoing_edge_array_size,counting_bits(sockets_num),
                    [&](EdgeType<EdgeData> e){ return get_local_partition_id(e.dst);});


            outgoing_edges = new ULLONG[sockets_num];
            for(int i = 0; i < sockets_num; ++i)
                outgoing_edges[i] = g_offsets[i+1] - g_offsets[i];


            outgoing_adj_list = new EdgeType<EdgeData> *[sockets_num];
            for(int s_i = 0; s_i < sockets_num; ++s_i)
                outgoing_adj_list[s_i] = (EdgeType<EdgeData> *) numa_alloc_onnode(sizeof(EdgeType<EdgeData>) * outgoing_edges[s_i], s_i);


#pragma omp parallel num_threads(threads_num)
            {
                int t_i = omp_get_thread_num();
                int s_i = get_socket_idx(t_i);
                int t_j = get_socket_offset(t_i);
                int range = (outgoing_edges[s_i] + threads_per_socket - 1) / threads_per_socket;
                UINT begin = t_j * range;
                UINT end = std::min<uint>(outgoing_edges[s_i], (t_j + 1) * range);
                EdgeType<EdgeData>* adj_list = outgoing_adj_list[s_i];
                UINT off = g_offsets[s_i];

                for(uint i = begin; i < end; ++i){
                    adj_list[i] = inter_outgoing_edge_array[off+i];
                }
            }


            // outgoing_adj_bitmap[s_i]->get_bit(u) means there is an edge which dst vertex belongs to socket s_i and src vertex is u
            // outgoing_adj_list[s_i] consists all dst vertex and additional weight data in socket s_i;
            // outgoing_adj_index[s_i][v_i] indicate the starting edge range which src is v_i

            // allocate and initialize
            outgoing_adj_index = new ULLONG *[sockets_num];
            outgoing_adj_bitmap = new BitmapType *[sockets_num];
            for (int s_i = 0; s_i < sockets_num; s_i++) {
                outgoing_adj_bitmap[s_i] = new BitmapType(vertices_num);
                outgoing_adj_bitmap[s_i]->clear();
                outgoing_adj_index[s_i] = (ULLONG *) numa_alloc_onnode(sizeof(ULLONG) * (vertices_num + 1), s_i);
            }

#pragma omp parallel num_threads(sockets_num)
            {
                // get socket id
                int s_i = omp_get_thread_num();
                numa_run_on_node(s_i);
                ULLONG local_edges_num = outgoing_edges[s_i];
                EdgeType<EdgeData> * adj_list = outgoing_adj_list[s_i];
                BitmapType* adj_bitmap = outgoing_adj_bitmap[s_i];
                ULLONG * adj_index = outgoing_adj_index[s_i];


                // construct adj_bitmap and adj_index array
                // adj_bitmap can set simply when encounter a new vertex
                // adj_index is more complex, we need record two things, start point and end point
                // start point can directly record when encounter a new vertex
                // end point should adjacent to start point, however not every vertices appear in this socket edge
                // so we use pre + 1 to guarantee adjacent property

                // deal with start point
                UINT pre = adj_list[0].src;
                adj_bitmap->set_bit(pre);
                adj_index[pre] = 0;

                // loop entire array
                for(UINT i = 1; i < local_edges_num; ++i){
                    UINT src = adj_list[i].src;

                    // when vertex id is changed, we encounter a new vertex
                    // we should set new vertex bit
                    if(pre != src){
                        adj_bitmap->set_bit(src);
                        // record starting index in edge array
                        adj_index[src] = i;
                        // noted, src may not continual, so we should used pre_src+1
                        adj_index[pre + 1] = i;
                        pre = src;
                    }
                }
                // also need last index
                adj_index[pre+1] = local_edges_num;

            }
            return;
        }

        // load a directed graph from path
        void load_directed(std::string filepath, UINT vertices_num) {
#ifdef PRINT_DEBUG_MESSAGES
            if (partition_idx==0) {
                printf("load directed graph file\n");
            }
#endif

            preload_graph_file(filepath,vertices_num,false);

            prep_time = 0;
            prep_time -= MPI_Wtime();
            split_graph();

            prep_time += MPI_Wtime();

        }


        // process vertices_num
        template<typename R>
            R process_vertices(std::function<R(UINT)> process, BitmapType *active) {
                double stream_time = 0;
                stream_time -= MPI_Wtime();

                R reducer = 0;
                size_t basic_chunk = 64;
                for (int t_i = 0; t_i < threads_num; t_i++) {
                    int s_i = get_socket_idx(t_i);
                    int s_j = get_socket_offset(t_i);
                    UINT partition_size = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
                    thread_state[t_i]->curr =
                        local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
                    thread_state[t_i]->end = local_partition_offset[s_i] +
                        partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
                    if (s_j == threads_per_socket - 1) {
                        thread_state[t_i]->end = local_partition_offset[s_i + 1];
                    }
                    thread_state[t_i]->status = WORKING;
                }
#pragma omp parallel num_threads(threads_num) reduction(+:reducer)
                {
                    R local_reducer = 0;
                    int thread_id = omp_get_thread_num();
                    while (true) {
                        UINT v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                        if (v_i >= thread_state[thread_id]->end) break;
                        unsigned long word = active->data[WORD_OFFSET(v_i)];
                        while (word != 0) {
                            if (word & 1) {
                                local_reducer += process(v_i);
                            }
                            v_i++;
                            word = word >> 1;
                        }
                    }
                    thread_state[thread_id]->status = STEALING;
                    for (int t_offset = 1; t_offset < threads_num; t_offset++) {
                        int t_i = (thread_id + t_offset) % threads_num;
                        while (thread_state[t_i]->status != STEALING) {
                            UINT v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                            if (v_i >= thread_state[t_i]->end) continue;
                            unsigned long word = active->data[WORD_OFFSET(v_i)];
                            while (word != 0) {
                                if (word & 1) {
                                    local_reducer += process(v_i);
                                }
                                v_i++;
                                word = word >> 1;
                            }
                        }
                    }
                    reducer += local_reducer;
                }
                R global_reducer;
                MPI_Datatype dt = get_mpi_data_type<R>();
                MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
                stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
                if (partition_idx==0) {
                    printf("process_vertices took %lf (s)\n", stream_time);
                }
#endif
                return global_reducer;
            }

        // flush thread local send buffer to global send buffer
        // put local_send_buffer data to send_buffer on current_send_part_idx with same socket_idx of thread_idx
        template<typename M>
            void flush_local_send_buffer(int thread_idx) {
                int socket_idx = get_socket_idx(thread_idx);
                int pos = __sync_fetch_and_add(&send_buffer[current_send_part_idx][socket_idx]->count,
                        local_send_buffer[thread_idx]->count);
                memcpy(send_buffer[current_send_part_idx][socket_idx]->data + sizeof(MsgUnit<M>) * pos,
                        local_send_buffer[thread_idx]->data,
                        sizeof(MsgUnit<M>) * local_send_buffer[thread_idx]->count);

                local_send_buffer[thread_idx]->count = 0;
            }

        // emit a message to a vertex's master (dense) / mirror (sparse)
        // message will firstly put into thread's local_send_buffer
        // if local_send_buffer is full, flush local_send_buffer to global send_buffer
        template<typename M>
            void emit(UINT vtx, M msg) {
                int thread_idx = omp_get_thread_num();
                MsgUnit<M> *buffer = (MsgUnit<M> *) local_send_buffer[thread_idx]->data;
                buffer[local_send_buffer[thread_idx]->count].vertex = vtx;
                buffer[local_send_buffer[thread_idx]->count].msg_data = msg;
                local_send_buffer[thread_idx]->count += 1;
                if (local_send_buffer[thread_idx]->count == local_send_buffer_limit) {
                    flush_local_send_buffer<M>(thread_idx);
                }
            }

        // process read_edges_num
        template<typename R, typename M>
            R process_edges(std::function<void(UINT)> sparse_signal,
                    std::function<R(UINT, M, AdjacentListType<EdgeData>)> sparse_slot,
                    std::function<void(UINT, AdjacentListType<EdgeData>)> dense_signal,
                    std::function<R(UINT, M)> dense_slot, BitmapType *active, BitmapType *dense_selective = nullptr) {
                double stream_time = 0;
                stream_time -= MPI_Wtime();
                size_t basic_chunk = 64;
                R reducer = 0;

                // determined 
                if(use_undirected){
                    UINT active_edges = process_vertices<UINT>(
                        [&](UINT vtx) {
                            return (UINT) out_degree[vtx];
                        },
                        active
                    );
                    sparse = (active_edges < edges / 20);
                }
                else if(use_iteration){
                    sparse=0;
                }
                else{
                    sparse=1;
                }

                for (int t_i = 0; t_i < threads_num; t_i++) {
                    local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
                    local_send_buffer[t_i]->count = 0;
                }

                if (sparse) {
                    for (int i = 0; i < partitions_num; i++) {
                        for (int s_i = 0; s_i < sockets_num; s_i++) {
                            recv_buffer[i][s_i]->resize(
                                    sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets_num);
                            send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets_num);
                            send_buffer[i][s_i]->count = 0;
                            recv_buffer[i][s_i]->count = 0;
                        }
                    }
                } else {
                    for (int i = 0; i < partitions_num; i++) {
                        for (int s_i = 0; s_i < sockets_num; s_i++) {
                            recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets_num);
                            send_buffer[i][s_i]->resize(
                                    sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets_num);
                            send_buffer[i][s_i]->count = 0;
                            recv_buffer[i][s_i]->count = 0;
                        }
                    }
                }

                if (sparse) {
#pragma omp parallel for
                    for (int t_i = 0; t_i < threads_num; t_i++) {
                        int s_i = get_socket_idx(t_i);
                        assert(numa_run_on_node(s_i) == 0);
                    }

                    current_send_part_idx = partition_idx;
#pragma omp parallel for num_threads(threads_num)
                    for (UINT begin_v_i = partition_offset[partition_idx];
                            begin_v_i < partition_offset[partition_idx + 1]; begin_v_i += basic_chunk) {
                        UINT v_i = begin_v_i;
                        unsigned long word = active->data[WORD_OFFSET(v_i)];
                        while (word != 0) {
                            if (word & 1) {
                                sparse_signal(v_i);
                            }
                            v_i++;
                            word = word >> 1;
                        }
                    }
#pragma omp parallel for
                    for (int t_i = 0; t_i < threads_num; t_i++) {
                        flush_local_send_buffer<M>(t_i);
                    }

                    // non-blocking send operation
                    MPI_Request *request = new MPI_Request;

                    for (int step = 1; step < partitions_num; step++) {
                        // send operation
                        int i = (partition_idx - step + partitions_num) % partitions_num;
                        for (int s_i = 0; s_i < sockets_num; s_i++) {
                            exe_data += send_buffer[partition_idx][s_i]->count;
                            MPI_Isend(send_buffer[partition_idx][s_i]->data,
                                    sizeof(MsgUnit<M>) * send_buffer[partition_idx][s_i]->count, MPI_CHAR, i, PassMessage,
                                    MPI_COMM_WORLD, request);
                            MPI_Request_free(request);
                        }

                        i = (partition_idx + step) % partitions_num;
                        for (int s_i = 0; s_i < sockets_num; s_i++) {
                            MPI_Status recv_status;
                            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
                            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
                            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
                        }
                    }



                    for (int step = 0; step < partitions_num; step++) {
                        int i = (partition_idx + step) % partitions_num;
                        MessageBuffer **used_buffer;
                        if (i == partition_idx) {
                            used_buffer = send_buffer[i];
                        } else {
                            used_buffer = recv_buffer[i];
                        }

                        for(int s_n = 0; s_n < sockets_num; ++s_n){
                            MsgUnit<M> *buffer = (MsgUnit<M> *) used_buffer[s_n]->data;
                            size_t buffer_size = used_buffer[s_n]->count;
                            for (int t_i = 0; t_i < threads_num; t_i++) {
                                //int s_i = get_socket_idx(t_i);
                                int s_j = get_socket_offset(t_i);
                                int worker_num = threads_per_socket;
                                UINT partition_size = buffer_size;
                                thread_state[t_i]->curr = partition_size / worker_num / basic_chunk * basic_chunk * s_j;
                                thread_state[t_i]->end =
                                    partition_size / worker_num/ basic_chunk * basic_chunk * (s_j + 1);
                                if (s_j == worker_num - 1) {
                                    thread_state[t_i]->end = buffer_size;
                                }
                                thread_state[t_i]->status = WORKING;
                            }
#pragma omp parallel num_threads(threads_num) reduction(+:reducer)
                            {
                                R local_reducer = 0;
                                int thread_id = omp_get_thread_num();
                                int s_i = get_socket_idx(thread_id);
                                while (true) {
                                    UINT b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                                    if (b_i >= thread_state[thread_id]->end) break;
                                    UINT begin_b_i = b_i;
                                    UINT end_b_i = b_i + basic_chunk;
                                    if (end_b_i > thread_state[thread_id]->end) {
                                        end_b_i = thread_state[thread_id]->end;
                                    }
                                    for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                        UINT v_i = buffer[b_i].vertex;
                                        M msg_data = buffer[b_i].msg_data;
                                        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                                            local_reducer += sparse_slot(v_i, msg_data, AdjacentListType<EdgeData>(
                                                        outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i],
                                                        outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i + 1]));
                                        }
                                    }
                                }
                                thread_state[thread_id]->status = STEALING;
                                for (int t_offset = 1; t_offset < threads_num; t_offset++) {
                                    int t_i = (thread_id + t_offset) % (threads_num);
                                    if (thread_state[t_i]->status == STEALING) continue;
                                    while (true) {
                                        UINT b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                                        if (b_i >= thread_state[t_i]->end) break;
                                        UINT begin_b_i = b_i;
                                        UINT end_b_i = b_i + basic_chunk;
                                        if (end_b_i > thread_state[t_i]->end) {
                                            end_b_i = thread_state[t_i]->end;
                                        }
                                        int s_i = get_socket_idx(t_i);
                                        for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                            UINT v_i = buffer[b_i].vertex;
                                            M msg_data = buffer[b_i].msg_data;
                                            if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                                                local_reducer += sparse_slot(v_i, msg_data, AdjacentListType<EdgeData>(
                                                            outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i],
                                                            outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i + 1]));
                                            }
                                        }
                                    }
                                }
                                reducer += local_reducer;
                            }
                        }
                    }
                } else {
#pragma omp parallel for
                    for (int t_i = 0; t_i < threads_num; t_i++) {
                        int s_i = get_socket_idx(t_i);
                        assert(numa_run_on_node(s_i) == 0);
                    }

                    MPI_Request* request = new MPI_Request;
                    MPI_Status recv_status;

                    if (dense_selective != nullptr && partitions_num > 1) {

                        for (int step = 1; step < partitions_num; step++) {

                            int recipient_id = (partition_idx + step) % partitions_num;
                            exe_data += owned_vertices / 64 * sizeof(unsigned long);
                            MPI_Isend(dense_selective->data + WORD_OFFSET(partition_offset[partition_idx]),
                                    owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD, request);
                            MPI_Request_free(request);

                            int sender_id = (partition_idx - step + partitions_num) % partitions_num;
                            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]),
                                    (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64,
                                    MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                    }


                    current_send_part_idx = partition_idx;
                    for (int step = 0; step < partitions_num; step++) {
                        current_send_part_idx = (current_send_part_idx + 1) % partitions_num;
                        int i = current_send_part_idx;
                        for (int t_i = 0; t_i < threads_num; t_i++)
                            *thread_state[t_i] = tuned_chunks_dense[i][t_i];
#pragma omp parallel num_threads(threads_num)
                        {
                            int thread_id = omp_get_thread_num();
                            int s_i = get_socket_idx(thread_id);
                            UINT final_p_v_i = thread_state[thread_id]->end;
                            while (true) {
                                UINT begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                                if (begin_p_v_i >= final_p_v_i) break;
                                UINT end_p_v_i = begin_p_v_i + basic_chunk;
                                if (end_p_v_i > final_p_v_i) {
                                    end_p_v_i = final_p_v_i;
                                }
                                for (UINT p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                                    UINT v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                                    dense_signal(v_i, AdjacentListType<EdgeData>(
                                                incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index,
                                                incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
                                }
                            }
                            thread_state[thread_id]->status = STEALING;
                            for (int t_offset = 1; t_offset < threads_num; t_offset++) {
                                int t_i = (thread_id + t_offset) % (threads_num);
                                int s_i = get_socket_idx(t_i);
                                while (thread_state[t_i]->status != STEALING) {
                                    UINT begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                                    if (begin_p_v_i >= thread_state[t_i]->end) break;
                                    UINT end_p_v_i = begin_p_v_i + basic_chunk;
                                    if (end_p_v_i > thread_state[t_i]->end) {
                                        end_p_v_i = thread_state[t_i]->end;
                                    }
                                    for (UINT p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                                        UINT v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                                        dense_signal(v_i, AdjacentListType<EdgeData>(
                                                    incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index,
                                                    incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
                                    }
                                }
                            }
                        }
#pragma omp parallel for
                        for (int t_i = 0; t_i < threads_num; t_i++)
                            flush_local_send_buffer<M>(t_i);
                        if(i != partition_idx){
                            //printf("part %d send to %d\n",partition_idx,i);
                            for (int s_i = 0; s_i < sockets_num; s_i++) {
                                exe_data += sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count;
                                MPI_Isend(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR,
                                i, PassMessage, MPI_COMM_WORLD, request);
                                MPI_Request_free(request);
                            }

                            i = (partition_idx - step - 1  + partitions_num) % partitions_num;
                            //printf("part %d wait recv from %d\n",partition_idx,i);
                            for(int s_i = 0; s_i < sockets_num;  ++s_i){
                                MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
                                MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
                                MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
                            }
                        }
                    }


                    // processing recv_queue
                    for (int step = 0; step < partitions_num; step++) {
                        int i = (partition_idx - step - 1  + partitions_num) % partitions_num;
                        MessageBuffer **used_buffer;
                        if (i == partition_idx) {
                            used_buffer = send_buffer[i];
                        } else {
                            used_buffer = recv_buffer[i];
                        }

                        for (int t_i = 0; t_i < threads_num; t_i++) {
                            int s_i = get_socket_idx(t_i);
                            int s_j = get_socket_offset(t_i);
                            int worker_num = threads_per_socket;
                            UINT partition_size = used_buffer[s_i]->count;
                            thread_state[t_i]->curr = partition_size / worker_num / basic_chunk * basic_chunk * s_j;
                            thread_state[t_i]->end =
                                partition_size / worker_num / basic_chunk * basic_chunk * (s_j + 1);
                            if (s_j == worker_num - 1) {
                                thread_state[t_i]->end = used_buffer[s_i]->count;
                            }
                            thread_state[t_i]->status = WORKING;
                        }
#pragma omp parallel num_threads(threads_num) reduction(+:reducer)
                        {
                            R local_reducer = 0;
                            int thread_id = omp_get_thread_num();
                            int s_i = get_socket_idx(thread_id);
                            MsgUnit<M> *buffer = (MsgUnit<M> *) used_buffer[s_i]->data;
                            while (true) {
                                UINT b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                                if (b_i >= thread_state[thread_id]->end) break;
                                UINT begin_b_i = b_i;
                                UINT end_b_i = b_i + basic_chunk;
                                if (end_b_i > thread_state[thread_id]->end) {
                                    end_b_i = thread_state[thread_id]->end;
                                }
                                for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                    UINT v_i = buffer[b_i].vertex;
                                    M msg_data = buffer[b_i].msg_data;
                                    local_reducer += dense_slot(v_i, msg_data);
                                }
                            }
                            thread_state[thread_id]->status = STEALING;
                            reducer += local_reducer;
                        }
                    }
                }

                R global_reducer;
                MPI_Datatype dt = get_mpi_data_type<R>();
                MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
                stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
                if (partition_idx==0) {
                    if(sparse)
                        printf("sparse, process_edges: %lf s \n",stream_time);
                    else
                        printf("dense,  process_edges: %lf s \n",stream_time);
                }
#endif
                return global_reducer;
            }

};

#endif

