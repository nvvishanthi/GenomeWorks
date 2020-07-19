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
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
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

// define constants. Taken from application parser
#define INDEX_SIZE 30
#define KMER_SIZE  15
#define WINDOWS_SIZE 10

// prints overlaps to stdout
void print_overlaps(const std::vector<Overlap>& overlaps,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser)
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
                                                     overlaps[i].num_residues_ * KMER_SIZE, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
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

// Finish function signature
std::unique_ptr<Index> initialize_batch(const std::shared_ptr<const io::FastaParser> parser,
                                        DefaultDeviceAllocator allocator)
{
    // Get device information.
    int32_t device_count = 0;
    GW_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    //Init();

    // group reads into indices. For small inputs, there will only be 1 index
    std::vector<IndexDescriptor> index_descriptors  = group_reads_into_indices(*parser,
                                                                               INDEX_SIZE * 1'000'000);
    // Put all reads into a single index.
    std::unique_ptr<Index> index = Index::create_index(allocator,
                                                       *parser,
                                                       index_descriptors.front().first_read(),
                                                       index_descriptors.front().first_read() + index_descriptors.front().number_of_reads(),
                                                       KMER_SIZE,
                                                       WINDOWS_SIZE);
    return std::move(index);
}

void process_batch(Index& query_index,
                   Index& target_index,
                   const std::shared_ptr<const io::FastaParser> query_parser,
                   const std::shared_ptr<const io::FastaParser> target_parser,
                   DefaultDeviceAllocator allocator,
                   bool print)
{
    // find anchors & find overlaps
    auto matcher = Matcher::create_matcher(allocator,
                                           query_index,
                                           target_index);

    // Output vector of overlaps
    std::vector<Overlap> overlaps;
    OverlapperTriggered overlapper(allocator);

    // Constant for overlapper
    int32_t min_residues                    = 3;
    int32_t min_overlap_len                 = 250;
    int32_t min_bases_per_residue           = 1000;
    float min_overlap_fraction              = 0.8;

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
    if (print)
    {
        print_overlaps(overlaps, *query_parser, *target_parser);
    }

    return;
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
    if (help)
    {
        std::cout << "CUDA Mapper API sample program. Runs minimizer-based approximate mapping" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudamapper [-p] [-h]" << std::endl;
        std::cout << "-p : Print the overlaps to stdout" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // Load FASTA/FASTQ file. Assume all-to-all
    const std::string query_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta";
    const std::string target_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta";

    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);

    // create FASTA parser
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;
    query_parser = io::create_kseq_fasta_parser(query_file, KMER_SIZE + WINDOWS_SIZE -1);   // defaults taken from application parser
    target_parser = query_parser;   // assume all to all

    std::unique_ptr<Index> query_index  = initialize_batch(query_parser, allocator);
    std::unique_ptr<Index> target_index = initialize_batch(target_parser, allocator);

    process_batch(*query_index, *target_index, query_parser, target_parser, allocator, print);

    return 0;
}