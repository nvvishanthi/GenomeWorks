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

#pragma once

#include <mutex>
#include <vector>

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{
class FastaParser;
}; // namespace io

namespace cudamapper
{

/// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
/// \param overlaps vector of overlap objects
/// \param cigars CIGAR strings
/// \param query_parser needed for read names and lenghts
/// \param target_parser needed for read names and lenghts
/// \param kmer_size minimizer kmer size
/// \param write_output_mutex mutex that enables exclusive access to output stream
void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigars,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               int32_t kmer_size,
               std::mutex& write_output_mutex);

/// \brief Given a string s, produce its kmers (length <kmer-length>) and return them as a vector of strings.
/// \param s A string sequence to kmerize.
/// \param kmer_size A kmer length to use for producing kmers.
/// \param stride The number of bases to skip when selecting kmers (most often, this should be equal to 1).
/// \return A vector of strings containing the kmers (of length kmer_length) of s. If s is shorter than the kmer size, return s.
std::vector<gw_string_view_t> split_into_kmers(const gw_string_view_t& s, std::int32_t kmer_size, std::int32_t stride);

/// \brief Given two sorted vectors of comparable types, return a size_t count of the number of shared elements.
/// Duplicates are counted the number of times they appear (i.e., two vectors of ten identical elements would
/// return a shared count of 10).
/// \param a A sorted vector of elements. These must be comparable (i.e., they must implement the == and < operators) and sorted in ascending order.
/// \param b A sorted vector of elements. These must be comparable with those in a and sorted in ascending order.
/// \return The number of elements the two sets have in common, including repeated elements, as a std::size_t.
template <typename T>
std::size_t count_shared_elements(const std::vector<T>& a, const std::vector<T>& b);

/// \brief Given two sequences 'a' and 'b', calculate an estimate of their similarity
/// Calculates the approximate nucleotide identity (or "similarity")
/// estimated from the Jaccard index of the kmers of strings a and b.
/// Note: This function assumes that a and b are on the same strand; you may need to
/// reverse-complement one of the strings if testing similarity on strings from different
/// strands.
/// \param a A C++ string
/// \param b A C++ string
/// \param kmer_size The kmer length to use for estimating similarity.
/// \param stride The number of bases to stride between kmers.
/// \return The estimated Jaccard index as a float.
float sequence_jaccard_similarity(const gw_string_view_t& a, const gw_string_view_t& b, std::int32_t kmer_size, std::int32_t stride);

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
