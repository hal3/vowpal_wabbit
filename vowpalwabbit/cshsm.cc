/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#include <errno.h>

#include "reductions.h"
#include "v_hashmap.h"
#include "label_dictionary.h"
#include "vw.h"
#include "gd.h" // GD::foreach_feature() needed in subtract_example()
#include "vw_exception.h"
#include "rand48.h"

using namespace std;
using namespace LEARNER;
namespace CS=COST_SENSITIVE;

struct cshsm
{ uint32_t K;
  uint32_t root;   // top layer branching factor
  uint32_t leaf;   // max bottom layer branching factor
  polyprediction* pred_root, * pred_leaf;
  bool classificationesque;
  float initial;
  set<uint32_t> update_bottom;
  v_array<float> top_costs;
};

/***

    suppose we have a tree: [[a b] [c d]]
    call the branches l and r

    if costs are (a:0 b:2) (c:0.5 d:1.9)
    then what are the appropriate costs for l:? and r:?

    the OPTIMIST  would say l:0 r:0.5
    the PESSIMIST would say l:2 r:1.9
    the REALIST   would say l:x r:y
      where x is cost of h(a vs b) and y is cost of h(c vs d)
      for current hypothesis h

    pessimist seems like a bad idea. realist is going to be
    expensive. so optimist seems reasonable.

    next question: which leaf classifiers do we update?

    the OPTIMIST would say only l because i'll assume i eventually get l vs r right
    that seems way too optimistic

    maybe update minimum cost *and* current prediction as a tradeoff
    between REALIST and OPTIMIST.

***/

template <bool is_learn>
void predict_or_learn(cshsm& hsm, LEARNER::base_learner& base, example& ec) {
  CS::label ld = ec.l.cs;
  
  //  uint32_t trueL = ld.label - 1;
  //  uint32_t true0 = trueL / hsm.leaf;

  base.multipredict(ec, 0, hsm.root, hsm.pred_root, false);
  //cerr << "hsm.predR ="; for (size_t i=0; i<hsm.root; i++) cerr << ' ' << hsm.pred[i].scalar; //cerr << endl;
  uint32_t pred0 = 0;
  for (uint32_t i=1; i<hsm.root; i++)
    if (hsm.pred_root[i].scalar < hsm.pred_root[pred0].scalar)
      pred0 = i;

  uint32_t top = hsm.leaf; // min(hsm.leaf, hsm.k - hsm.leaf * pred0 + 1);
  
  base.multipredict(ec, hsm.root + hsm.leaf * pred0, top, hsm.pred_leaf, false);
  uint32_t pred1 = 0;
  //cerr << "hsm.pred" << pred0 << " ="; for (size_t i=0; i<top; i++) cerr << ' ' << hsm.pred[i].scalar; cerr << endl;
  for (uint32_t i=1; i<top; i++)
    if (hsm.pred_leaf[i].scalar < hsm.pred_leaf[pred1].scalar)
      pred1 = i;

  if (is_learn) {
    // at the bottom level, we update (all minima) \cup (current prediction)
    hsm.update_bottom.clear();  // which ids
    for (size_t i=0; i<hsm.root; i++)
      hsm.top_costs[i] = FLT_MAX;

    hsm.update_bottom.insert(pred0);
    for (CS::wclass& wc : ld.costs)
    { size_t j = (wc.class_index-1) / hsm.leaf;
      if (wc.x <= 0.)
        hsm.update_bottom.insert(j);
      if (wc.x < hsm.top_costs[j])
        hsm.top_costs[j] = wc.x;
    }
    //cerr << "hsm.update_bottom = "; for (auto x : hsm.update_bottom) cerr << x << ' '; cerr << endl;
    //cerr << "hsm.top_costs     = "; for (auto x : hsm.top_costs) cerr << x << ' '; cerr << endl;

    ec.l.simple = { 0.f, 1.f, 0.f };
    for (size_t i=0; i<hsm.root; i++)
      if (hsm.top_costs[i] < FLT_MAX)
      { ec.l.simple.label = hsm.top_costs[i] * 2. - 1.;
        ec.partial_prediction = hsm.pred_root[i].scalar;
        ec.pred.scalar = ec.partial_prediction;
        //cerr << "learn0(" << i << ")" << endl;
        base.update(ec, i);
      }

    for (CS::wclass& wc : ld.costs)
    { size_t j = (wc.class_index-1) / hsm.leaf;
      if (hsm.update_bottom.find(j) != hsm.update_bottom.end())
      { ec.l.simple.label = wc.x * 2. - 1.;
        //cerr << "learn1(" << hsm.root + wc.class_index - 1 << ")" << endl;
        if (j == pred0)
        { ec.partial_prediction = hsm.pred_leaf[(wc.class_index-1) % hsm.leaf].scalar;
          ec.pred.scalar = ec.partial_prediction;
          base.update(ec, hsm.root + wc.class_index - 1);
        } else
          base.learn(ec, hsm.root + wc.class_index - 1);
      }
    }
    
    /*
    set<uint32_t> trueL;
    set<uint32_t> true0;
    for (CS::wclass& wc : ld.costs)
      if (wc.x <= 0.)
      { trueL.insert(wc.class_index-1);
        true0.insert((wc.class_index-1) / hsm.leaf);
        //cerr << "  trueL <- " << wc.class_index-1 << "\ttrue0 <- " << ((wc.class_index-1) / hsm.leaf) << endl;
      }

    ec.l.simple = { 0.f, 1., 0.f };
    for (uint32_t i=0; i<hsm.root; i++) {
      ec.l.simple.label = (true0.find(i) != true0.end()) ? -1. : 1.;
      //cerr << "  baseR[" << i << "].learn(" << ec.l.simple.label << ")" << endl;
      base.learn(ec, i);
    }

    //top = min(hsm.leaf, hsm.k - hsm.leaf * true0 + 1);
    for (uint32_t t0 : true0)
      for (uint32_t i=0; i<top; i++) {
        uint32_t j = t0 * hsm.leaf + i;
        ec.l.simple.label = (trueL.find(j) != trueL.end()) ? -1. : 1.;
        //cerr << "  base" << i << "[" << hsm.root + j << "].learn(" << ec.l.simple.label << ")" << endl;
        base.learn(ec, hsm.root + j);
      }
    */
  }

  ec.pred.multiclass = pred0 * hsm.leaf + pred1 + 1;
  ec.l.cs = ld;
}

void finish_example(vw& all, cshsm&, example& ec)
{ //output_example(all, ec);
  VW::finish_example(all, &ec);
}

void finish(cshsm& c)
{ free(c.pred_root);
  free(c.pred_leaf);
  c.top_costs.delete_v();
}


base_learner* cshsm_setup(vw& all)
{ if (missing_option<size_t, true>(all, "cshsm", "One-against-all multiclass with <k> costs via hierarchical softmax"))
    return nullptr;

  cshsm& c = calloc_or_throw<cshsm>();

  c.K = (uint32_t)all.vm["cshsm"].as<size_t>();

  new_options(all, "CSHSM Options")
      ("classificationesque", "switch CSHSM into classification mode")
      ("hsm_branch", po::value<uint32_t>(&c.root)->default_value(2 * (int)ceilf(sqrt((float)c.K))), "branching factor at root")
      ("initial_cost", po::value<float>(&c.initial)->default_value(0.), "set the initial prediction value (default 0.; 1. makes cshsm pessimistic)");
  add_options(all);

  c.leaf = (uint32_t)ceilf((float)c.K / (float)c.root);
  
  //c.pred = calloc_or_throw<polyprediction>(max(c.root, c.leaf));
  c.pred_root = calloc_or_throw<polyprediction>(c.root);
  c.pred_leaf = calloc_or_throw<polyprediction>(c.leaf);
  c.classificationesque = all.vm.count("classificationesque") > 0;

  c.top_costs = v_init<float>();
  for (size_t i=0; i<c.root; i++)
    c.top_costs.push_back(FLT_MAX);
  
  learner<cshsm>& l = init_learner(&c,
                                   setup_base(all),
                                   predict_or_learn<true>,
                                   predict_or_learn<false>,
                                   (uint32_t)(c.root * (c.leaf + 1)));
  all.p->lp = CS::cs_label;
  l.set_finish_example(finish_example);
  l.set_finish(finish);
  base_learner* b = make_base(l);
  all.cost_sensitive = b;
  return b;
}
