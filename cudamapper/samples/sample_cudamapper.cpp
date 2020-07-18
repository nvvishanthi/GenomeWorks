/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claraparabricks/genomeworks/cudamapper/cudamapper.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp> // may not need this
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include "../src/index_descriptor.hpp"
#include "../src/overlapper_triggered.hpp"
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

//     std::unique_ptr<Index> batch = /*something*/; // create the batch here

//     return std::move(batch);
// }

void process_batch()
{
    return;
}

void print_overlaps(const std::vector<Overlap>& overlaps,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const int32_t kmer_size)
{
    const int64_t number_of_overlaps_to_print = get_size<int64_t>(overlaps);

    if (number_of_overlaps_to_print <= 0)
    {
        return;
    }

    // All overlaps are saved to a single vector of chars and that vector is then printed to output.

    // Allocate approximately 150 characters for each overlap which will be processed,
    // if more characters are needed buffer will be reallocated.
    std::vector<char> buffer(150 * number_of_overlaps_to_print);
    // characters written buffer so far
    int64_t chars_in_buffer = 0;

    {
        for (int64_t i = 0; i < number_of_overlaps_to_print; ++i)
        {
            const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
            const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;
            // (over)estimate the number of character that are going to be needed
            // 150 is an overestimate of number of characters that are going to be needed for non-string values
            int32_t expected_chars = 150 + get_size<int32_t>(query_read_name) + get_size<int32_t>(target_read_name);

            // if there is not enough space in buffer reallocate
            if (get_size<int64_t>(buffer) - chars_in_buffer < expected_chars)
            {
                buffer.resize(buffer.size() * 2 + expected_chars);
            }
            // Add basic overlap information.
            const int32_t added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                     "%s\t%lu\t%i\t%i\t%c\t%s\t%lu\t%i\t%i\t%i\t%ld\t%i",
                                                     query_read_name.c_str(),
                                                     query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length(),
                                                     overlaps[i].query_start_position_in_read_,
                                                     overlaps[i].query_end_position_in_read_,
                                                     static_cast<unsigned char>(overlaps[i].relative_strand),
                                                     target_read_name.c_str(),
                                                     target_parser.get_sequence_by_id(overlaps[i].target_read_id_).seq.length(),
                                                     overlaps[i].target_start_position_in_read_,
                                                     overlaps[i].target_end_position_in_read_,
                                                     overlaps[i].num_residues_ * kmer_size, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
                                                     std::max(std::abs(static_cast<int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<int64_t>(overlaps[i].target_end_position_in_read_)),
                                                              std::abs(static_cast<int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                     255);
            chars_in_buffer += added_chars;

            // Add new line to demarcate new entry.
            buffer[chars_in_buffer] = '\n';
            ++chars_in_buffer;
        }
        buffer[chars_in_buffer] = '\0';
    }

    printf("%s", buffer.data());
}

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
    // BEGIN intialize_batch
    // std::unique_ptr<Index> batch = initialize_batch(); // fix this

    // create FASTA parser here
    uint32_t kmer_size    = 15;
    uint32_t windows_size = 10;
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;
    query_parser = io::create_kseq_fasta_parser(query_file, 1, false);   // defaults taken from application parser
    // assume all-to-all
    target_parser = query_parser;

    // group reads into indices

    // split indices into IndexDescriptors - these two pieces make up a single batch. This hsould be fine as.
    std::vector<IndexDescriptor> query_index_descriptors  = group_reads_into_indices(*query_parser,
                                                                                    30 * 1'000'000);
    std::vector<IndexDescriptor> target_index_descriptors = group_reads_into_indices(*target_parser,
                                                                                     30 * 1'000'000);


    // might need to duplicate the above to for the device.. maybe not?

    // END initialize_batch

    std::unique_ptr<Index> query_index  = Index::create_index(allocator,
                                                              *query_parser,
                                                              query_index_descriptors[0].first_read(),
                                                              query_index_descriptors[0].first_read() + query_index_descriptors[0].number_of_reads(),
                                                              kmer_size,
                                                              windows_size);   // leave other default arguments as-is
    std::unique_ptr<Index> target_index = Index::create_index(allocator,
                                                              *target_parser,
                                                              target_index_descriptors[0].first_read(),
                                                              target_index_descriptors[0].first_read() + target_index_descriptors[0].number_of_reads(),
                                                              kmer_size,
                                                              windows_size);   // leave other default arguments as-is

    // // TODO VI: Abstract the following to process_batch function?
    // // for each pair of indices, find anchors & find overlaps
    auto matcher = Matcher::create_matcher(allocator,
                                           *query_index,
                                           *target_index);

    //Overlapper overlapper(allocator);
    std::vector<Overlap> overlaps;
    OverlapperTriggered overlapper(allocator);

    int32_t min_residues                    = 3;     // r, recommended range: 1 - 10. Higher: more accurate. Lower: more sensitive
    int32_t min_overlap_len                 = 250;   // l, recommended range: 100 - 1000
    int32_t min_bases_per_residue           = 1000;  // b
    float min_overlap_fraction              = 0.8;   // z
    overlapper.get_overlaps(overlaps,
                            matcher->anchors(),
                            min_residues,
                            min_overlap_len,
                            min_bases_per_residue,
                            min_overlap_fraction);

    // reset the matcher
    matcher.reset(nullptr);

    // post process the overlaps
    Overlapper::post_process_overlaps(overlaps);
    // print the overlaps
    print_overlaps(overlaps, *query_parser, *target_parser, kmer_size);
    std::cout << "reached the end" << std::endl;

    return 0;
}