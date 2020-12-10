/*
* Copyright 2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "chainer_utils.cuh"

#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <thrust/reduce.h>
#include <fstream>
#include <sstream>
#include <cstdlib>

// Needed for accumulate - remove when ported to cuda
#include <numeric>
#include <limits>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
namespace chainerutils
{

#define BLOCK_COUNT 1792
#define BLOCK_SIZE 64

struct ConvertOverlapToNumResidues : public thrust::unary_function<Overlap, int32_t>
{
    __host__ __device__ int32_t operator()(const Overlap& o) const
    {
        return o.num_residues_;
    }
};

__host__ __device__ Overlap create_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    bool is_negative_strand               = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b)
{
    return a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
}

__device__ bool operator==(const QueryReadID& a, const QueryReadID& b)
{
    return a.query_read_id_ == b.query_read_id_;
}

__device__ Overlap create_simple_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    bool is_negative_strand               = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

Overlap create_simple_overlap_cpu(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    const bool is_negative_strand         = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

// TODO VI: this may have some thread overwrite issues, as well as some problems
// uninitialized variables in overlap struct.
// This also has an upper bound on how many anchors we actually process. If num_anchors is greater
// than 1792 * 32, we never actually process that anchor
__global__ void backtrace_anchors_to_overlaps_old(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride  = blockDim.x * gridDim.x;
    for (int i = tid; i < n_anchors; i += stride)
    {
        double score = scores[i];
        if (score >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor         = anchors[first_index];
            overlaps[i]                 = create_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i]          = false;
            overlaps[i]                 = create_overlap(empty_anchor(), empty_anchor(), 1);
        }
    }
}

// new backtrace function
__global__ void backtrace_anchors_to_overlaps(const Anchor* const anchors,
                                              Overlap* const overlaps,
                                              int32_t* const overlap_terminal_anchors,
                                              const float* const scores,
                                              bool* const max_select_mask,
                                              const int32_t* const predecessors,
                                              const int64_t n_anchors,
                                              const float min_score)
{
    const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;

    for (int i = thread_id; i < n_anchors; i += stride)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor         = anchors[first_index];
            overlap_terminal_anchors[i] = i;
            overlaps[i]                 = create_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i]          = false;
            overlap_terminal_anchors[i] = -1;
            overlaps[i]                 = create_overlap(empty_anchor(), empty_anchor(), 1);
        }
    }
}

void allocate_anchor_chains(const device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int64_t& num_total_anchors,
                            DefaultDeviceAllocator allocator,
                            cudaStream_t cuda_stream)
{
    auto thrust_exec_policy = thrust::cuda::par(allocator).on(cuda_stream);
    thrust::plus<int32_t> sum_op;
    num_total_anchors = thrust::transform_reduce(thrust_exec_policy,
                                                 overlaps.begin(),
                                                 overlaps.end(),
                                                 ConvertOverlapToNumResidues(),
                                                 0,
                                                 sum_op);

    unrolled_anchor_chains.clear_and_resize(num_total_anchors);
    anchor_chain_starts.clear_and_resize(overlaps.size());

    thrust::transform_exclusive_scan(thrust_exec_policy,
                                     overlaps.begin(),
                                     overlaps.end(),
                                     anchor_chain_starts.data(),
                                     ConvertOverlapToNumResidues(),
                                     0,
                                     sum_op);
}

__global__ void output_overlap_chains_by_backtrace(const Overlap* const overlaps,
                                                   const Anchor* const anchors,
                                                   const bool* const select_mask,
                                                   const int32_t* const predecessors,
                                                   const int32_t* const chain_terminators,
                                                   int32_t* const anchor_chains,
                                                   int32_t* const anchor_chain_starts,
                                                   const int32_t num_overlaps,
                                                   const bool check_mask)
{
    const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;

    // Processes one overlap per iteration,
    // "i" corresponds to an overlap
    for (int i = thread_id; i < num_overlaps; i += stride)
    {

        if (!check_mask || (check_mask & select_mask[i]))
        {
            int32_t anchor_chain_index = 0;
            // As chaining proceeds backwards (i.e., it's a backtrace),
            // we need to fill the new anchor chain array in in reverse order.
            int32_t index = chain_terminators[i];
            while (index != -1)
            {
                anchor_chains[anchor_chain_starts[i] + (overlaps[i].num_residues_ - anchor_chain_index - 1)] = index;
                index                                                                                        = predecessors[index];
                ++anchor_chain_index;
            }
        }
    }
}

__global__ void output_overlap_chains_by_RLE(const Overlap* const overlaps,
                                             const Anchor* const anchors,
                                             const int32_t* const chain_starts,
                                             const int32_t* const chain_lengths,
                                             int32_t* const anchor_chains,
                                             int32_t* const anchor_chain_starts,
                                             const uint32_t num_overlaps)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;
    for (uint32_t i = thread_id; i < num_overlaps; i += stride)
    {
        int32_t chain_start  = chain_starts[i];
        int32_t chain_length = chain_lengths[i];
        for (int32_t index = chain_start; index < chain_start + chain_length; ++index)
        {
            anchor_chains[index] = index;
        }
    }
}

__global__ void backtrace_anchors_to_overlaps_debug(const Anchor* anchors,
                                                    Overlap* overlaps,
                                                    double* scores,
                                                    bool* max_select_mask,
                                                    int32_t* predecessors,
                                                    const int32_t n_anchors,
                                                    const int32_t min_score)
{
    int32_t end = n_anchors - 1;
    for (int32_t i = end; i >= 0; i--)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}

void backtrace_anchors_to_overlaps_cpu(const Anchor* anchors,
                                       Overlap* overlaps,
                                       double* scores,
                                       bool* max_select_mask,
                                       int32_t* predecessors,
                                       const int32_t n_anchors,
                                       const int32_t min_score)
{
    int32_t end = n_anchors - 1;
    for (int32_t i = end; i >= 0; i--)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_simple_overlap_cpu(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts)
{
    std::int32_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride     = blockDim.x * gridDim.x;
    for (int i = d_tid; i < n_starts; i += stride)
    {
        ends[d_tid] = starts[d_tid] + lengths[d_tid];
    }
}

// we take each query that is encoded as length (ie there are 4 queries that have the same id)
// so we take the (how many queries are in that section) and divide it by the size of the tile to get the
// number of tiles for that query
__global__ void calculate_tiles_per_read(const std::int32_t* lengths,
                                         const int32_t num_reads,
                                         const int32_t tile_size,
                                         std::int32_t* tiles_per_read)
{
    int32_t d_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_thread_id < num_reads)
    {
        // if the number of queries in that section are not evenly divisible, we need an extra block
        // to accomadate for the leftovers
        int32_t n_integer_blocks    = lengths[d_thread_id] / tile_size;
        int32_t remainder           = lengths[d_thread_id] % tile_size;
        tiles_per_read[d_thread_id] = remainder == 0 ? n_integer_blocks : n_integer_blocks + 1;
    }
}

__global__ void calculate_tile_starts(const std::int32_t* query_starts,
                                      const std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      const int32_t num_queries,
                                      const std::int32_t* tiles_per_query_up_to_point)
{
    int32_t d_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_thread_id < num_queries)
    {
        // for each tile, we look up the query it corresponds to and offset it by the which tile in the query
        // we're at multiplied by the total size of the tile
        for (int i = 0; i < tiles_per_query[d_thread_id]; i++)
        {
            //finds the offset in the ragged array and offsets to find start of "next" sub array
            tile_starts[tiles_per_query_up_to_point[d_thread_id] + i] = query_starts[d_thread_id] + (i * tile_size);
        }
    }
}

std::vector<std::string> tokenize_line(std::string line, const char delim = '\t')
{
    std::vector<std::string> tokens;
    std::stringstream st(line);
    std::string tmp;
    while (std::getline(st, tmp, delim))
    {
        tokens.push_back(tmp);
    }
    return tokens;
}

std::vector<seed_debug_entry> read_minimap2_seed_chains(const char* const seed_file_name)
{

    std::vector<seed_debug_entry> seeds;
    int32_t q_id = 0;

    std::ifstream seed_file_stream;
    seed_file_stream.open(seed_file_name);
    std::string line;
    if (seed_file_stream.good())
    {

        while (std::getline(seed_file_stream, line))
        {
            if (line[0] == '[')
            {
                continue;
            }
            std::vector<std::string> tokens = tokenize_line(line, '\t');
            if (tokens[0] == "QR")
            {
                // Create a new seed_debug_entry
                seed_debug_entry current;
                current.query_id     = tokens[1];
                current.query_int_id = q_id++;
                // Skip the next RS line, as we can't do anything with it yet.
                std::getline(seed_file_stream, line);
                // Retrieve the first seed line
                std::getline(seed_file_stream, line);
                // Parse all the seeds
                while (line.substr(0, 2) == "SD")
                {
                    // process seed line to generate and store a corresponding Anchor
                    tokens = tokenize_line(line, '\t');
                    current.add_seed(tokens);

                    std::getline(seed_file_stream, line);
                }
                while (line.substr(0, 2) == "CN")
                {
                    // process the chain line and store it
                    tokens = tokenize_line(line, '\t');
                    current.add_chain_entry(tokens);
                    std::getline(seed_file_stream, line);
                }
                seeds.push_back(current);
            }
        }

        // Clear the current chain
    }

    seed_file_stream.close();
    return seeds;
}

void encode_query_locations_from_anchors(const Anchor* anchors,
                                         int32_t n_anchors,
                                         device_buffer<int32_t>& query_starts,
                                         device_buffer<int32_t>& query_lengths,
                                         device_buffer<int32_t>& query_ends,
                                         int32_t& n_queries,
                                         DefaultDeviceAllocator& _allocator,
                                         cudaStream_t& _cuda_stream)
{
    AnchorToQueryReadIDOp anchor_to_read_op;
    // This takes anchors and outputs and converts the anchors to QueryReadID types (references)
    cub::TransformInputIterator<QueryReadID, AnchorToQueryReadIDOp, const Anchor*> d_queries(anchors, anchor_to_read_op);
    // create buffer of size number of anchors
    device_buffer<QueryReadID> d_query_read_ids(n_anchors, _allocator, _cuda_stream);
    // vector of number of read ids...? I don't know what this is for
    // This is used to store the length of the encoded sequence
    device_buffer<int32_t> d_num_query_read_ids(1, _allocator, _cuda_stream);

    // don't know what this is for yet
    // this is for internal use for run length encoding
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(),
                                       d_num_query_read_ids.data(),
                                       n_anchors,
                                       _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_read_ids.data(),
                                       n_anchors,
                                       _cuda_stream);
    // this is just the "length" of the encoded sequence
    n_queries          = cudautils::get_value_from_device(d_num_query_read_ids.data(), _cuda_stream);
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(), // this is the vector of encoded lengths
                                  query_starts.data(),  // at this point, this vector is empty
                                  n_queries,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    // this gives us the number of queries up to that point. eg How many query starts we have at each index
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(), // this is the vector of encoded lengths
                                  query_starts.data(),
                                  n_queries,
                                  _cuda_stream);

    // paper uses the ends and finds the beginnings with x - w + 1, are we converting to that here?
    // TODO VI: I'm not entirely sure what this is for? I think we want to change the read query
    // (defined by [query_start, query_start + query_length] to [query_end - query_length + 1, query_end])
    // The above () is NOT true
    convert_offsets_to_ends<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(query_starts.data(),  // this gives how many starts at each index
                                                                          query_lengths.data(), // this is the vector of encoded lengths
                                                                          query_ends.data(),
                                                                          n_queries);
}

void encode_anchor_query_target_pairs(const Anchor* anchors,
                                      int32_t n_anchors,
                                      int32_t tile_size,
                                      device_buffer<int32_t>& query_target_starts,
                                      device_buffer<int32_t>& query_target_lengths,
                                      device_buffer<int32_t>& query_target_ends,
                                      device_buffer<int32_t>& tiles_per_read,
                                      int32_t& n_query_target_pairs,
                                      int32_t& n_qt_tiles,
                                      DefaultDeviceAllocator& _allocator,
                                      cudaStream_t& _cuda_stream,
                                      int32_t block_size)
{
    AnchorToQueryTargetPairOp qt_pair_op;
    cub::TransformInputIterator<QueryTargetPair, AnchorToQueryTargetPairOp, const Anchor*> d_query_target_pairs(anchors, qt_pair_op);
    device_buffer<QueryTargetPair> d_qt_pairs(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_target_pairs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_anchors,
                                       _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_target_pairs.data(),
                                       n_anchors,
                                       _cuda_stream);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(),
                                                                                                      query_target_lengths.data(),
                                                                                                      query_target_ends.data(),
                                                                                                      n_query_target_pairs);

    if (tile_size > 0)
    {
        calculate_tiles_per_read<<<(n_query_target_pairs / block_size) + 1, 32, 0, _cuda_stream>>>(query_target_starts.data(), n_query_target_pairs, tile_size, tiles_per_read.data());
        device_buffer<int32_t> d_n_qt_tiles(1, _allocator, _cuda_stream);

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage,
                               temp_storage_bytes,
                               tiles_per_read.data(),
                               d_n_qt_tiles.data(),
                               n_query_target_pairs,
                               _cuda_stream);
        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();
        cub::DeviceReduce::Sum(d_temp_storage,
                               temp_storage_bytes,
                               tiles_per_read.data(),
                               d_n_qt_tiles.data(),
                               n_query_target_pairs,
                               _cuda_stream);
        n_qt_tiles = cudautils::get_value_from_device(d_n_qt_tiles.data(), _cuda_stream);
    }
}

void encode_overlap_query_target_pairs(Overlap* overlaps,
                                       int32_t n_overlaps,
                                       device_buffer<int32_t>& query_target_starts,
                                       device_buffer<int32_t>& query_target_lengths,
                                       device_buffer<int32_t>& query_target_ends,
                                       int32_t& n_query_target_pairs,
                                       DefaultDeviceAllocator& _allocator,
                                       cudaStream_t& _cuda_stream,
                                       int32_t block_size)
{
    OverlapToQueryTargetPairOp qt_pair_op;
    cub::TransformInputIterator<QueryTargetPair, OverlapToQueryTargetPairOp, Overlap*> d_query_target_pairs(overlaps, qt_pair_op);
    device_buffer<QueryTargetPair> d_qt_pairs(n_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_target_pairs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps,
                                       _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps,
                                       _cuda_stream);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(), query_target_lengths.data(), query_target_ends.data(), n_query_target_pairs);
}

__global__ void initialize_mask(bool* mask, const int32_t n_values, bool val)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < n_values; i += grid_stride)
    {
        mask[i] = val;
    }
}

__global__ void initialize_array(int32_t* array, const int32_t num_values, int32_t value)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < num_values; i += grid_stride)
    {
        array[i] = value;
    }
}

__global__ void initialize_array(double* array, const int32_t num_values, double value)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < num_values; i += grid_stride)
    {
        array[i] = value;
    }
}

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks