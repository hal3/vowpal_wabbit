/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include "search_test_ldfnonldf.h"
#include "vw.h"

using namespace std;

namespace TestLDFNonLDFTask
{
Search::search_task task = { "ldfnonldf", run, initialize, nullptr, nullptr, nullptr };
namespace CS=COST_SENSITIVE;
  
void initialize(Search::search& sch, size_t& num_actions, po::variables_map& /*vm*/)
{ sch.set_options( Search::AUTO_CONDITION_FEATURES  |    // automatically add history features to our examples, please
                   Search::AUTO_HAMMING_LOSS        |    // please just use hamming loss on individual predictions -- we won't declare loss
                   Search::EXAMPLES_DONT_CHANGE     |    // we don't do any internal example munging
                   Search::IS_MIXED_LDF             |    // we have multiple learners, some are ldf
                   0);
  sch.ldf_alloc(num_actions);
  sch.set_num_learners( { false, true } ); // 2 learners, first is non-ldf, second is ldf
}

// this is totally bogus for the example -- you'd never actually do this!
void my_update_example_indicies(Search::search& sch, bool audit, example* ec, uint64_t mult_amount, uint64_t plus_amount)
{ size_t ss = sch.get_stride_shift();
  for (features& fs : *ec)
    for (feature_index& idx : fs.indicies)
      idx = (((idx >> ss) * mult_amount) + plus_amount) << ss;
}
  
void run(Search::search& sch, vector<example*>& ec)
{ action num_actions = sch.ldf_count();
  Search::predictor P(sch, (ptag)0);
  //cerr << "<<<<<<<<<< LDFnonLDF::run >>>>>>>>>>" << endl;

  for (ptag i=0; i<ec.size(); i++)
  { //cerr << ">>>>>> i = " << i << endl;
    // on even examples, do non-ldf; on odd examples, do ldf
    action prediction;
    action oracle = ec[i]->l.multi.label;
    if ((i % 2) == 0)
      prediction = P.set_learner_id(0).set_tag((ptag)i+1).set_input(*ec[i]).set_oracle(oracle).set_condition_range((ptag)i, sch.get_history_length(), 'p').predict();
    else
    { // ldf
      for (size_t a=0; a<num_actions; a++)
      { if (true || sch.predictNeedsExample())   // we can skip this work if `predict` won't actually use the example data
        { VW::copy_example_data(false, sch.ldf_example(a), ec[i]);  // copy but leave label alone!
          // now, offset it appropriately for the action id
          my_update_example_indicies(sch, true, sch.ldf_example(a), 28904713, 4832917 * (uint64_t)a);
        }
        // regardless of whether the example is needed or not, the class info is needed
        sch.ldf_set_label(a, a+1, 0.);
      }

      action pred_id = P.set_learner_id(1).set_tag((ptag)(i+1)).set_input(sch.ldf_example(), num_actions).set_oracle(oracle-1).set_condition_range(i, sch.get_history_length(), 'p').predict();
      prediction = pred_id+1;  // or sch.ldf_get_label(pred_id)
    }

    if (sch.output().good())
      sch.output() << sch.pretty_label((uint32_t)prediction) << ' ';
  }
}
}

