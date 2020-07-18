/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Add includes here (overlapper, index, matcher, io)

#include <claraparabricks/genomeworks/cudamapper/cudamapper.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp> // may not need this
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <cudamapper_file_location.hpp>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/cudamapper/matcher.hpp>
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

using namespace  claraparabricks::genomeworks;
using namespace  claraparabricks::genomeworks::cudamapper;

// Finish function signature
// Note that BatchofIndices is defined in an internal header... might not be able to use here
// std::unique_ptr<Index> initialize_batch(/*fill in function signature here*/)
// {
//         // Get device information.
//     int32_t device_count = 0;
//     GW_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
//     assert(device_count > 0);

//     // TODO VI: I don't think I need this
//     // size_t total = 0, free = 0;
//     // cudaSetDevice(0); // Using first GPU for sample.
//     // cudaMemGetInfo(&free, &total);

//     Init();

//     const int32_t device_id = 0;
//     cudaStream_t stream     = 0;

//     std::unique_ptr<BatchOfIndices> batch = /*something*/; // create the batch here

//     return std::move(batch);
// }

int main(int argc, char** argv)
{
    // parse command line options
    int c      = 0;
    bool help  = false;
    bool print = false;

    while ((c = getopt(argc, argv, "hp")) != -1)
    {
        switch (c)
        {
        case 'p':
            print = true;
            break;
        case 'h':
            help = true;
            break;
        }
    }

    // print help string
    // TODO VI: fix this to reflect CUDA Mapper args
    if (help)
    {
        std::cout << "CUDA Mapper API sample program. Runs minimizer-based approximate mapping" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudamapper [-h]" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // Load FASTA/FASTQ file. Hardcoded for now, do we want to allow for any file to be passed in?
    const std::string query_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta";
    const std::string target_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta"; // What does this need to be?

    // Group reads into index
    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);


    // abstract the batchs setup into initialize_batch()
    // Maybe the wrong type?
    // std::unique_ptr<Index> batch = initialize_batch(); // fix this

    // create FASTA parser here
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;
    query_parser = io::create_kseq_fasta_parser(query_file, 15 + 10 - 1);   // defaults taken from application parser
    // assume all-to-all
    target_parser = query_parser;

    // group reads into indices
    // Split work into batches
    // std::vector<BatchOfIndices> batches_of_indices_vect = generate_batches_of_indices(10,
    //                                                                                   5,
    //                                                                                   10,
    //                                                                                   5,
    //                                                                                   query_parser,
    //                                                                                   target_parser,
    //                                                                                   30 * 1'000'000,        // value was in MB
    //                                                                                   30 * 1'000'000, // value was in MB
    //                                                                                   true); //all to all mode

    // generate indices 
    std::unique_ptr<Index> query_index = Index::create_index(allocator, *query_parser, 0, 0, 15, 10); // leave other default arguments as-is
    std::unique_ptr<Index> target_index  = Index::create_index(allocator, *target_parser, 0, 0, 15, 10); // leave other default arguments as-is


    // // single thread, single stream
    // // we only need a single stream here, bc I suppose only 1 GPU is needed? 

    // // TODO VI: Abstract the following to process_batch function?
    // // for each pair of indices, find anchors & find overlaps
    auto matcher = Matcher::create_matcher(allocator,
                                            *query_index,
                                            *target_index);

    std::cout << "reached the end" << std::endl; 
    //Overlapper overlapper(allocator);
    std::vector<Overlap> overlaps;
    Overlapper::get_overlaps(overlaps,
                             matcher->anchors(),
                             3, // min_residues default value
                             250,
                             1000,
                             0.8);

    // reset the matcher
    // overlapper.get_overlaps(overlaps,
    //                         matcher->anchors(),
    //                         application_parameters.min_residues,
    //                         application_parameters.min_overlap_len,
    //                         application_parameters.min_bases_per_residue,
    //                         application_parameters.min_overlap_fraction);

    return 0;
}