/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#pragma once
#include "search.h"

namespace TestLDFNonLDFTask
{
void initialize(Search::search&, size_t&, po::variables_map&);
void run(Search::search&, std::vector<example*>&);
extern Search::search_task task;
}
