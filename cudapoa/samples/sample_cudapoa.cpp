/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../benchmarks/common/utils.hpp"

#include <file_location.hpp>
#include <claragenomics/cudapoa/cudapoa.hpp>
#include <claragenomics/cudapoa/batch.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/genomeutils.hpp>

#include "spoa/spoa.hpp"

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <unistd.h>
#include <random>
#include <iomanip>
#include <omp.h>

using namespace claragenomics;
using namespace claragenomics::cudapoa;

std::unique_ptr<Batch> initialize_batch(bool msa, const BatchSize& batch_size, bool banded_alignment = false)
{
    // Get device information.
    int32_t device_count = 0;
    CGA_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    size_t total = 0, free = 0;
    cudaSetDevice(0); // Using first GPU for sample.
    cudaMemGetInfo(&free, &total);

    // Initialize internal logging framework.
    Init();

    // Initialize CUDAPOA batch object for batched processing of POAs on the GPU.
    const int32_t device_id      = 0;
    cudaStream_t stream          = 0;
    size_t mem_per_batch         = 0.9 * free; // Using 90% of GPU available memory for CUDAPOA batch.
    const int32_t mismatch_score = -6, gap_score = -8, match_score = 8;

    std::unique_ptr<Batch> batch = create_batch(device_id,
                                                stream,
                                                mem_per_batch,
                                                msa ? OutputType::msa : OutputType::consensus,
                                                batch_size,
                                                gap_score,
                                                mismatch_score,
                                                match_score,
                                                banded_alignment);

    return std::move(batch);
}

void process_batch(Batch* batch, bool msa_flag, bool print,
                   std::vector<std::vector<std::string>>& msa,
                   std::vector<std::string>& consensus,
                   std::vector<std::vector<uint16_t>>& coverage)
{
    batch->generate_poa();

    StatusType status = StatusType::success;
    if (msa_flag)
    {
        // Grab MSA results for all POA groups in batch.
        std::vector<std::vector<std::string>> msa_local; // MSA per group
        std::vector<StatusType> output_status;           // Status of MSA generation per group

        status = batch->get_msa(msa_local, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate MSA for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(msa_local); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating  MSA for POA group " << g << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    for (const auto& alignment : msa_local[g])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }
        }

        msa.insert(msa.end(), msa_local.begin(), msa_local.end());
    }
    else
    {
        // Grab consensus results for all POA groups in batch
        std::vector<std::string> consensus_local;
        std::vector<std::vector<uint16_t>> coverage_local;
        std::vector<StatusType> output_status; // Status of consensus generation per group

        status = batch->get_consensus(consensus_local, coverage_local, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate consensus for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(consensus_local); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating consensus for POA group " << g << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    std::cout << consensus_local[g] << std::endl;
                }
            }
        }

        consensus.insert(consensus.end(), consensus_local.begin(), consensus_local.end());
        coverage.insert(coverage.end(), coverage_local.begin(), coverage_local.end());
    }
}

void spoa_compute(const std::vector<std::vector<std::string>>& groups,
                  const int32_t start_id, const int32_t end_id, const int32_t number_of_threads,
                  bool msa_flag, bool print,
                  std::vector<std::vector<std::string>>& msa,
                  std::vector<std::string>& consensus,
                  std::vector<std::vector<uint32_t>>& coverage)
{
    spoa::AlignmentType atype = spoa::AlignmentType::kNW;
    int match_score           = 8;
    int mismatch_score        = -6;
    int gap_score             = -8;

    if (msa_flag)
    {

        // Grab MSA results for all groups within the range
        std::vector<std::vector<std::string>> msa_local(end_id - start_id); // MSA per group

#pragma omp parallel for num_threads(number_of_threads)
        for (int32_t g = start_id; g < end_id; g++)
        {
            auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
            auto graph            = spoa::createGraph();
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            graph->generate_multiple_sequence_alignment(msa_local[g - start_id]);
            graph->clear();
        }

        if (print)
        {
            std::cout << std::endl;
            for (int32_t i = 0; i < get_size(msa_local); i++)
            {
                {
                    for (const auto& alignment : msa_local[i])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }
        }

        msa.insert(msa.end(), msa_local.begin(), msa_local.end());
    }
    else
    {
        // Grab consensus results for all POA groups within the range
        std::vector<std::string> consensus_local(end_id - start_id);          // Consensus string for each POA group
        std::vector<std::vector<uint32_t>> coverage_local(end_id - start_id); // Per base coverage for each consensus

#pragma omp parallel for num_threads(number_of_threads)
        for (int32_t g = start_id; g < end_id; g++)
        {
            auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
            auto graph            = spoa::createGraph();
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            consensus_local[g - start_id] = graph->generate_consensus(coverage_local[g - start_id]);
            graph->clear();
        }

        if (print)
        {
            std::cout << std::endl;
            for (int32_t i = 0; i < get_size(consensus_local); i++)
            {
                std::cout << consensus_local[i] << std::endl;
            }
        }

        consensus.insert(consensus.end(), consensus_local.begin(), consensus_local.end());
        coverage.insert(coverage.end(), coverage_local.begin(), coverage_local.end());
    }
}

void generate_window_data(const std::string& input_file, const int number_of_windows, const int max_sequences_per_poa,
                          std::vector<std::vector<std::string>>& windows, BatchSize& batch_size)
{
    parse_window_data_file(windows, input_file, number_of_windows); // Generate windows.
    assert(get_size(windows) > 0);

    int32_t max_read_length = 0;
    for (auto& window : windows)
    {
        int n = 0;
        for (auto& seq : window)
        {
            n++;
            max_read_length = std::max(max_read_length, get_size<int>(seq) + 1);
            if (n >= max_sequences_per_poa)
            {
                window.erase(window.begin() + n, window.end());
                break;
            }
        }
    }

    batch_size = BatchSize(max_read_length, max_sequences_per_poa);
}

// the following function can create simulated reads in an arbitrary number of windows with given sequence length and POA group size
void generate_simulated_reads(const uint32_t number_of_windows, const uint32_t sequence_size, const uint32_t max_sequences_per_poa,
                              std::vector<std::vector<std::string>>& windows, BatchSize& batch_size)
{
    constexpr uint32_t random_seed = 5827349;
    std::minstd_rand rng(random_seed);

    int32_t max_sequence_length = sequence_size + 1;

    // create variation_ranges used to generate some random variations in random positions
    std::vector<std::pair<int, int>> variation_ranges;
    uint32_t range_begin, range_end, range_width;
    const int32_t num_ranges = 10;
    uint32_t max_range_width = sequence_size / (2 * num_ranges);
    uint32_t step_size       = 2 * max_range_width;
    uint32_t start_idx       = 0;
    for (uint32_t i = 0; i < num_ranges; i++)
    {
        if (step_size == 0)
        {
            break;
        }

        std::uniform_int_distribution<uint32_t> random_pos(start_idx, start_idx + step_size);
        range_begin = random_pos(rng);
        std::uniform_int_distribution<uint32_t> random_width(0, max_range_width);
        range_width = random_width(rng);
        if ((range_begin + range_width) > (start_idx + step_size))
        {
            range_end   = start_idx + step_size;
            range_begin = range_end - range_width;
        }
        else
        {
            range_end = range_begin + range_width;
        }
        variation_ranges.push_back(std::pair<int, int>(range_begin, range_end));

        start_idx += step_size;
    }

    std::vector<std::string> long_reads(max_sequences_per_poa);
    for (uint32_t w = 0; w < number_of_windows; w++)
    {
        long_reads[0] = claragenomics::genomeutils::generate_random_genome(sequence_size, rng);
        for (uint32_t i = 1; i < max_sequences_per_poa; i++)
        {
            long_reads[i]       = claragenomics::genomeutils::generate_random_sequence(long_reads[0], rng, sequence_size, sequence_size, sequence_size, &variation_ranges);
            max_sequence_length = max_sequence_length > get_size(long_reads[i]) ? max_sequence_length : get_size(long_reads[i]) + 1;
        }
        // add long reads as one window
        windows.push_back(long_reads);
    }

    // Define upper limits for sequence size, graph size ....
    batch_size = BatchSize(max_sequence_length, max_sequences_per_poa);
}

int main(int argc, char** argv)
{
    // Process options
    int c            = 0;
    bool msa_flag    = false;
    bool long_read   = false;
    bool help        = false;
    bool print       = false;
    bool print_graph = false;
    bool benchmark   = false;
    bool banded      = false;
    bool bonito      = true;
    // following parameters are used in benchmarking only
    uint32_t number_of_windows = 0;
    uint32_t sequence_size     = 0;
    uint32_t group_size        = 0;
    int32_t number_of_threads  = 0;
    // benchmark mode 0: runs only cudaPOA, 1: runs only SPOA, 2: runs both (default)
    uint32_t benchmark_mode = 2;
    // an option to report detailed accuracy metrics per window as opposed to an average value
    uint32_t verbose = 1;

    while ((c = getopt(argc, argv, "mlhpgbBRW:S:t:N:M:V:")) != -1)
    {
        switch (c)
        {
        case 'm':
            msa_flag = true;
            break;
        case 'l':
            long_read = true;
            break;
        case 'p':
            print = true;
            break;
        case 'g':
            print_graph = true;
            break;
        case 'b':
            benchmark = true;
            break;
        case 'B':
            banded = true;
            break;
        case 'R':
            bonito = false;
            break;
        case 'V':
            verbose = atoi(optarg);
            break;
        case 'W':
            number_of_windows = atoi(optarg);
            break;
        case 'S':
            sequence_size = atoi(optarg);
            break;
        case 't':
            number_of_threads = atoi(optarg);
            break;
        case 'N':
            group_size = atoi(optarg);
            break;
        case 'M':
            benchmark_mode = atoi(optarg);
            break;
        case 'h':
            help = true;
            break;
        }
    }

    if (help)
    {
        std::cout << "CUDAPOA API sample program. Runs consensus or MSA generation on pre-canned data." << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudapoa [-m] [-h]" << std::endl;
        std::cout << "-m : Generate MSA (if not provided, generates consensus by default)" << std::endl;
        std::cout << "-l : Perform long-read sample (if not provided, will run short-read sample by default)" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-g : Print POA graph in dot format, this option is only for long-read sample" << std::endl;
        std::cout << "-b : Benchmark against SPOA" << std::endl;
        std::cout << "-t : Maximum number of threads used in SPOA benchmarking (if not provided, will default to number of logical CPUs)" << std::endl;
        std::cout << "-B : cudaPOA is computed as banded" << std::endl;
        std::cout << "-W : Number of total windows used in benchmarking" << std::endl;
        std::cout << "-S : Maximum sequence length in benchmarking" << std::endl;
        std::cout << "-N : Number of sequences per POA group" << std::endl;
        std::cout << "-R : For long-read use simulated random sequences (default uses data from bonito basecaller for Oxford Nanopore reads)" << std::endl;
        std::cout << "-M : 0, 1, [2]. Only used in benchmark mode: -M 0, runs only cudaPOA, -M 1, runs only SPOA, -M 2, default, runs both" << std::endl;
        std::cout << "-V : 0, [1]. Verbose mode, only used in benchmark mode. -V 1, default, will output details per window. -V 0 only reports average metrics among windows." << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // if not defined as input args, set default values for benchmarking parameters
    number_of_windows = number_of_windows == 0 ? (long_read ? 10 : 1000) : number_of_windows;
    sequence_size     = sequence_size == 0 ? (long_read ? 10000 : 1024) : sequence_size;
    group_size        = group_size == 0 ? (long_read ? 6 : 100) : group_size;
    number_of_threads = number_of_threads == 0 ? (int32_t)omp_get_num_procs() : number_of_threads;

    if (benchmark && number_of_threads < 1)
    {
        std::cerr << "invalid number of threads. continuing benchmarking with 1 thread, see argument -- 't'" << std::endl;
        number_of_threads = 1;
    }

    if (!long_read)
    {
        if (sequence_size != 1024)
            std::cerr << "for short read samples, input maximum sequence length is ignored, see argument -- 'S'" << std::endl;
    }

    // Load input data. Each POA group is represented as a vector of strings. The sample
    // data for short reads has many such POA groups to process, hence the data is loaded into a vector
    // of vector of strings. Long read sample creates one POA group.
    std::vector<std::vector<std::string>> windows;

    // Define upper limits for sequence size, graph size ....
    BatchSize batch_size;

    if (benchmark)
    {
        if (long_read)
        {
            if (bonito)
            {
                const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-bonito.txt";
                generate_window_data(input_file, number_of_windows, group_size, windows, batch_size);
            }
            else
            {
                generate_simulated_reads(number_of_windows, sequence_size, group_size, windows, batch_size);
            }
        }
        else
        {
            const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
            generate_window_data(input_file, number_of_windows, group_size, windows, batch_size);
        }
    }
    else
    {
        if (long_read)
        {
            const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-bonito.txt";
            generate_window_data(input_file, 8, 6, windows, batch_size);
        }
        else
        {
            const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
            generate_window_data(input_file, 1000, 100, windows, batch_size);
        }
    }

    // Initialize batch.
    std::unique_ptr<Batch> batch = initialize_batch(msa_flag, batch_size, banded);

    // Loop over all the POA groups, add them to the batch and process them.
    int32_t window_count = 0;
    // for benchmarking
    float cudapoa_time = 0.f;
    float spoa_time    = 0.f;
    ChronoTimer timer;

    // results vectors
    std::vector<std::vector<std::string>> msa_c;   // MSA per group, for cudapoa
    std::vector<std::string> consensus_c;          // Consensus string for each POA group, for cudapoa
    std::vector<std::vector<uint16_t>> coverage_c; // Per base coverage for each consensus, for cudapoa
    std::vector<std::vector<std::string>> msa_s;   // MSA per group, for spoa
    std::vector<std::string> consensus_s;          // Consensus string for each POA group, for spoa
    std::vector<std::vector<uint32_t>> coverage_s; // Per base coverage for each consensus, for spoa

    for (int32_t i = 0; i < get_size(windows);)
    {
        const std::vector<std::string>& window = windows[i];

        Group poa_group;
        // Create a new entry for each sequence and add to the group.
        for (const auto& seq : window)
        {
            Entry poa_entry{};
            poa_entry.seq     = seq.c_str();
            poa_entry.length  = seq.length();
            poa_entry.weights = nullptr;
            poa_group.push_back(poa_entry);
        }

        std::vector<StatusType> seq_status;
        StatusType status = batch->add_poa_group(seq_status, poa_group);

        // NOTE: If number of windows smaller than batch capacity, then run POA generation
        // once last window is added to batch.
        if (status == StatusType::exceeded_maximum_poas || status == StatusType::exceeded_batch_size || (i == get_size(windows) - 1))
        {
            if(batch->get_total_poas()>0)
            {
                // No more POA groups can be added to batch. Now process batch.
                if (benchmark)
                {
                    if (benchmark_mode != 1)
                    {
                        timer.start_timer();
                        process_batch(batch.get(), msa_flag, print, msa_c, consensus_c, coverage_c);
                        cudapoa_time += timer.stop_timer();
                    }

                    if (benchmark_mode != 0)
                    {
                        timer.start_timer();
                        spoa_compute(windows, window_count, window_count + batch->get_total_poas(), number_of_threads, msa_flag, print, msa_s, consensus_s, coverage_s);
                        spoa_time += timer.stop_timer();
                    }
                }
                else
                {
                    process_batch(batch.get(), msa_flag, print, msa_c, consensus_c, coverage_c);
                }

                if (print_graph && long_read)
                {
                    std::vector<DirectedGraph> graph;
                    std::vector<StatusType> graph_status;
                    batch->get_graphs(graph, graph_status);
                    for (auto& g : graph)
                    {
                        std::cout << g.serialize_to_dot() << std::endl;
                    }
                }

                // After MSA/consensus is generated for batch, reset batch to make room for next set of POA groups.
                batch->reset();

                // In case that number of windows is more than the capacity available on GPU, the for loop breaks into smaller number of windows.
                // if adding window i in batch->add_poa_group is not successful, it wont be processed in this iteration, therefore we print i-1
                // to account for the fact that window i was excluded at this round.
                if (status == StatusType::success)
                {
                    std::cout << "Processed windows " << window_count << " - " << i << std::endl;
                }
                else
                {
                    std::cout << "Processed windows " << window_count << " - " << i - 1 << std::endl;
                }
            }
            else
            {
                // the POA was too large to be added to the GPU, skip and move on
                std::cout << "Could not add POA group " << i << " to batch. Error code " << status << std::endl;
                if (benchmark)
                {
                    if (benchmark_mode != 1)
                        consensus_c.push_back("");
                    if (benchmark_mode != 0)
                        consensus_s.push_back("");
                }
                i++;
            }

            window_count = i;
        }

        if (status == StatusType::success)
        {
            // Check if all sequences in POA group wre added successfully.
            for (const auto& s : seq_status)
            {
                if (s == StatusType::exceeded_maximum_sequence_size)
                {
                    std::cerr << "Dropping sequence because sequence exceeded maximum size" << std::endl;
                }
            }
            i++;
        }

        if (status != StatusType::exceeded_maximum_poas && status != StatusType::exceeded_batch_size && status != StatusType::success)
        {
            std::cerr << "Could not add POA group " << i << " to batch. Error code " << status << std::endl;
            i++;
            if (benchmark)
            {
                if (benchmark_mode != 1)
                    consensus_c.push_back("");
                if (benchmark_mode != 0)
                    consensus_s.push_back("");
            }
        }
    }

    // print benchmarking report
    if (benchmark)
    {
        std::cerr << "\nbenchmark summary:\n";
        std::cerr << "=============================================================================================================\n";
        std::cerr << "Number of windows(W) " << std::left << std::setw(14) << std::fixed << number_of_windows;
        std::cerr << "Sequence length(S) " << std::left << std::setw(9) << std::fixed << sequence_size;
        std::cerr << "Number of sequences per window(N) " << std::left << std::setw(30) << group_size << std::endl;
        std::cerr << "Number of threads(t) " << std::left << std::setw(14) << std::fixed << number_of_threads;
        std::cerr << "Dataset " << std::left << std::setw(20) << std::fixed;
        if (long_read)
        {
            if (bonito)
                std::cerr << " bonito";
            else
                std::cerr << " simulated";
        }
        else
        {
            std::cerr << " short";
        }
        std::cerr << "Banded alignment for cudaPOA      ";
        if (banded)
            std::cerr << "ON\n";
        else
            std::cerr << "OFF\n";
        std::cerr << "-------------------------------------------------------------------------------------------------------------\n";
        std::cerr << "Compute time (sec):                cudaPOA " << std::left << std::setw(20);
        if (benchmark_mode == 1)
            std::cerr << "NA";
        else
            std::cerr << std::fixed << std::setprecision(2) << cudapoa_time;
        std::cerr << "SPOA ";
        if (benchmark_mode == 0)
            std::cerr << "NA" << std::endl;
        else
            std::cerr << std::fixed << std::setprecision(2) << spoa_time << std::endl;
        std::cerr << "-------------------------------------------------------------------------------------------------------------\n";
        int32_t number_of_bases = number_of_windows * sequence_size * group_size;
        std::cerr << "Expected performance (bases/sec) : cudaPOA ";
        std::cerr << std::left << std::setw(20) << std::fixed << std::setprecision(2) << std::scientific;
        if (benchmark_mode == 1)
            std::cerr << "NA";
        else
            std::cerr << (float)number_of_bases / cudapoa_time;
        if (benchmark_mode == 0)
            std::cerr << "SPOA NA" << std::endl;
        else
            std::cerr << "SPOA " << (float)number_of_bases / spoa_time << std::endl;
        int32_t actual_number_of_bases = 0;
        for (auto& w : windows)
        {
            for (auto& seq : w)
            {
                actual_number_of_bases += get_size(seq);
            }
        }
        float effective_perf_cupoa = (float)actual_number_of_bases / cudapoa_time;
        float effective_perf_spoa  = (float)actual_number_of_bases / spoa_time;
        std::cerr << "Effective performance (bases/sec): cudaPOA " << std::left << std::setw(20) << std::fixed << std::setprecision(2) << std::scientific;
        if (benchmark_mode == 1)
            std::cerr << "NA";
        else
            std::cerr << effective_perf_cupoa;
        if (benchmark_mode == 0)
            std::cerr << "SPOA NA";
        else
            std::cerr << "SPOA " << std::left << std::setw(20) << effective_perf_spoa;
        if (benchmark_mode == 2)
        {
            if (effective_perf_cupoa > effective_perf_spoa)
                std::cerr << "x" << std::fixed << std::setprecision(1) << effective_perf_cupoa / effective_perf_spoa << " faster";
            else
                std::cerr << "x" << std::fixed << std::setprecision(1) << effective_perf_spoa / effective_perf_cupoa << " slower";
        }
        std::cerr << "\n-------------------------------------------------------------------------------------------------------------\n";
        if (!msa_flag)
        {
            std::vector<int32_t> consensus_lengths_c(number_of_windows);
            std::vector<int32_t> consensus_lengths_s(number_of_windows);
            int32_t sum_consensus_length_c = 0;
            int32_t sum_consensus_length_s = 0;
            if (benchmark_mode != 1)
            {
                for (int w = 0; w < number_of_windows; w++)
                {
                    consensus_lengths_c[w] = consensus_c[w].length();
                    sum_consensus_length_c += consensus_lengths_c[w];
                }
            }
            if (benchmark_mode != 0)
            {
                for (int w = 0; w < number_of_windows; w++)
                {
                    consensus_lengths_s[w] = consensus_s[w].length();
                    sum_consensus_length_s += consensus_lengths_s[w];
                }
            }

            if (verbose == 1)
            {
                float similarity_percentage;
                for (int w = 0; w < number_of_windows; w++)
                {
                    int width = w < 9 ? 4 : w < 99 ? 3 : w < 999 ? 2 : 1;
                    std::cerr << "Consensus length for window " << w + 1 << std::left << std::setw(width) << ":"
                              << "  cudaPOA " << std::left << std::setw(20);
                    if (benchmark_mode == 1)
                        std::cerr << "NA";
                    else
                        std::cerr << consensus_lengths_c[w];
                    if (benchmark_mode == 0)
                        std::cerr << "SPOA NA";
                    else
                        std::cerr << "SPOA " << std::left << std::setw(20) << consensus_lengths_s[w];
                    if (benchmark_mode == 2)
                    {
                        similarity_percentage = 100.0f * (1.0f - (float)abs(consensus_lengths_c[w] - consensus_lengths_s[w]) / (float)consensus_lengths_s[w]);
                        std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << similarity_percentage << "% similar length";
                    }
                    std::cerr << std::endl;
                }
            }
            else
            {
                std::cerr << "Average consensus length:          cudaPOA " << std::left << std::setw(20);
                if (benchmark_mode == 1)
                    std::cerr << "NA";
                else
                    std::cerr << sum_consensus_length_c / number_of_windows;
                if (benchmark_mode == 0)
                    std::cerr << "SPOA NA";
                else
                    std::cerr << "SPOA " << std::left << std::setw(20) << sum_consensus_length_s / number_of_windows;
                if (benchmark_mode == 2)
                {
                    float length_similarity_percentage = 100.0f * (1.0f - (float)abs(sum_consensus_length_c - sum_consensus_length_s) / (float)sum_consensus_length_s);
                    std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << length_similarity_percentage << "% similar length";
                }
                std::cerr << std::endl;
            }
            if (verbose && benchmark_mode == 2)
            {
                // print accuracy metrics
                std::cerr << "-------------------------------------------------------------------------------------------------------------\n";

                std::vector<std::vector<std::string>> consensus_results(number_of_windows);
                for (int i = 0; i < number_of_windows; i++)
                {
                    consensus_results[i].push_back(consensus_c[i]);
                    consensus_results[i].push_back(consensus_s[i]);
                }

                msa_s.clear();
                spoa_compute(consensus_results, 0, number_of_windows, number_of_threads, true, false, msa_s, consensus_s, coverage_s);

                // print comparison details between cudaPOA and SPOA consensus per window
                for (int32_t g = 0; g < get_size(msa_s); g++)
                {
                    int32_t insert_cntr   = 0;
                    int32_t delete_cntr   = 0;
                    int32_t mismatch_cntr = 0;
                    int32_t identity_cntr = 0;

                    int width = g < 9 ? 6 : g < 99 ? 5 : g < 999 ? 4 : 3;
                    std::cerr << "Differences for window      " << g + 1 << std::left << std::setw(width) << ":";

                    if (msa_s[g].size() == 2)
                    {
                        const auto& target = msa_s[g][0];
                        const auto& query  = msa_s[g][1];
                        if (target.length() == query.length())
                        {
                            for (int32_t i = 0; i < target.length(); i++)
                            {
                                if (target[i] == '-')
                                    insert_cntr++;
                                else if (query[i] == '-')
                                    delete_cntr++;
                                else if (target[i] != query[i])
                                    mismatch_cntr++;
                                else /*target[i] == query[i]*/
                                    identity_cntr++;
                            }
                            float identity_percentage = 100.0f * (float)(identity_cntr) / (float)(std::min(consensus_lengths_s[g], consensus_lengths_c[g]));

                            std::cerr << "indels  " << std::left << std::setw(4) << insert_cntr << "/" << std::left << std::setw(15) << delete_cntr;
                            std::cerr << "mismatches " << std::left << std::setw(14) << mismatch_cntr;
                            std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << identity_percentage << "% identity " << std::endl;
                        }
                        else
                        {
                            std::cerr << "indels  " << std::left << std::setw(20) << "--------";
                            std::cerr << "mismatches " << std::left << std::setw(14) << "---";
                            std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << "NA"
                                      << "% identity " << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "indels  " << std::left << std::setw(20) << "--------";
                        std::cerr << "mismatches " << std::left << std::setw(14) << "---";
                        std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << "NA"
                                  << "% identity " << std::endl;
                    }
                }
            }
        }
        std::cerr << "=============================================================================================================\n\n";
    }

    return 0;
}
