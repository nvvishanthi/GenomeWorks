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

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{
/// \defgroup cudapoa CUDA POA package
/// Base docs for the cudapoa package (tbd)
/// \{

/// CUDA POA error type
enum StatusType
{
    success = 0,
    exceeded_maximum_poas,
    exceeded_maximum_sequence_size,
    exceeded_maximum_sequences_per_poa,
    node_count_exceeded_maximum_graph_size,
    edge_count_exceeded_maximum_graph_size,
    seq_len_exceeded_maximum_nodes_per_window,
    loop_count_exceeded_upper_bound,
    output_type_unavailable,
    generic_error
};

/// Initialize CUDA POA context.
StatusType Init();

/// OutputType - Enum for encoding type of output
enum OutputType
{
    consensus = 0x1,
    msa       = 0x1 << 1
};

/// \}
} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
