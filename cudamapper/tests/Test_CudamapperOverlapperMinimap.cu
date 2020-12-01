/*
* Copyright 2019-2020 NVIDIA CORPORATION.
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

#include "gtest/gtest.h"

#include "../src/overlapper_triggered.hpp"
#include "../src/overlapper_minimap.hpp"
#include "../src/chainer_utils.cuh"
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>
#include <claraparabricks/genomeworks/cudamapper/utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(TestCudamapperOverlapperMinimap, scoring_beyond_bandwidth)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    CudaStream cuda_stream           = make_cuda_stream();
    OverlapperMinimap overlapper(allocator, cuda_stream.get());

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1000;

    Anchor anchor2;
    anchor2.query_read_id_           = 1;
    anchor2.target_read_id_          = 2;
    anchor2.query_position_in_read_  = 200;
    anchor2.target_position_in_read_ = 1100;

    Anchor anchor3;
    anchor3.query_read_id_           = 1;
    anchor3.target_read_id_          = 2;
    anchor3.query_position_in_read_  = 300;
    anchor3.target_position_in_read_ = 1200;

    Anchor anchor4;
    anchor4.query_read_id_           = 1;
    anchor4.target_read_id_          = 2;
    anchor4.query_position_in_read_  = 400;
    anchor4.target_position_in_read_ = 1300;
}

TEST(TestCudamapperOverlapperMinimap, scoring_beyond_max_dist)
{
}

TEST(TestCudamapperOverlapperMinimap, scoring_same_anchor)
{
}

TEST(TestCudamapperOverlapperMinimap, scoring_normal_case)
{
}

TEST(TestCudamapperOverlapperMinimap, Chaining)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    CudaStream cuda_stream           = make_cuda_stream();
    OverlapperMinimap overlapper(allocator, cuda_stream.get());
}

TEST(TestCumapperOverlapperMinimap, ChainingWithMM2Seeds)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    CudaStream cuda_stream           = make_cuda_stream();

    // TODO: Add a permanent mm2 seed file. Should live in cudamapper/data
    std::string debug_file("../data/seeds_debug.txt");
    std::vector<chainerutils::seed_debug_entry> entries = chainerutils::read_minimap2_seed_chains(debug_file.c_str());

    for (auto const &entry : entries)
    {
        auto overlapper = make_unique<OverlapperMiniMap>(alloctor, cuda_stream.get());
        std::vector<Overlap> overlaps;

        overlapper->get_overlaps(overlaps,
                                entry.seeds,
                                false,
                                30,
                                30,
                                10,
                                0.9);
        
        Overlapper::post_process_overlaps(overlaps);
    }

}

} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks