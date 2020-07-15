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
#include <cudamapper_file_location.hpp>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/cudamapper/matcher.hpp>
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

using namespace  claraparabricks::genomeworks;

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
    const std::string input_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta";
    // create FASTA parser here

    // Group reads into index

    // for each pair of indices, find anchors & find overlaps


    return 0;
}