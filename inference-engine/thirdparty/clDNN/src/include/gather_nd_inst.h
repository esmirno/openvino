/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/gather_nd.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather_nd> : public typed_program_node_base<gather_nd> {
    using parent = typed_program_node_base<gather_nd>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using gather_nd_node = typed_program_node<gather_nd>;

template <>
class typed_primitive_inst<gather_nd> : public typed_primitive_inst_base<gather_nd> {
    using parent = typed_primitive_inst_base<gather_nd>;

public:
    static layout calc_output_layout(gather_nd_node const& node);
    static std::string to_string(gather_nd_node const& node);

public:
    typed_primitive_inst(network_impl& network, gather_nd_node const& desc);
};

using gather_nd_inst = typed_primitive_inst<gather_nd>;
}  // namespace cldnn
