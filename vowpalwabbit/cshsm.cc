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

#define disp(x) #x << '=' << x << ' '

struct cshsm
{ uint32_t K;
  uint32_t root;   // top layer branching factor
  uint32_t leaf;   // max bottom layer branching factor
  polyprediction* pred_root, * pred_leaf;
  bool classificationesque;
  float initial;
  set<uint32_t> update_bottom;
  v_array<float> top_costs;
  bool redundant;
  bool filter;
  uint32_t kbest;
  v_array< pair<float,uint32_t> > kbest_topk;
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

inline void set_label(cshsm& hsm, label_data& l, float cost, float min_cost, float max_cost)
{ if (hsm.classificationesque)
  { l.label  = (cost <= min_cost) ? -1.f : 1.f;
    l.weight = (cost <= min_cost) ? (max_cost - min_cost) : (cost - min_cost);
  } else
  { l.label = cost * 2. - 1.;
    l.weight = 1.;
  }
  l.initial = hsm.initial;
}

template <bool is_learn>
void predict_or_learn(cshsm& hsm, LEARNER::base_learner& base, example& ec) {
  CS::label ld = ec.l.cs;
  
  //  uint32_t trueL = ld.label - 1;
  //  uint32_t true0 = trueL / hsm.leaf;

  //for (CS::wclass& wc : ld.costs)
  //  if (wc.x == 0.) cerr << "label = " << wc.class_index << endl;
  
  base.multipredict(ec, 0, hsm.root, hsm.pred_root, false);
  //cerr << "hsm.predR ="; for (size_t i=0; i<hsm.root; i++) cerr << ' ' << hsm.pred_root[i].scalar; cerr << endl;
  uint32_t pred0 = 0;
  for (uint32_t i=0; i<hsm.root; i++)
  { if (hsm.pred_root[i].scalar < hsm.pred_root[pred0].scalar)
      pred0 = i;
    if (hsm.kbest > 0)
    { hsm.kbest_topk[i].first = hsm.pred_root[i].scalar;
      hsm.kbest_topk[i].second = i;
    }
  }
  if (hsm.kbest > 0)
    std::sort(hsm.kbest_topk.begin(), hsm.kbest_topk.begin() + hsm.root);

  uint32_t prediction = 0;
  
  uint32_t pred1 = 0;
  uint32_t pred_leaf_lo1 = 0, pred_leaf_hi1 = 0;
  uint32_t pred_leaf_lo2 = 0, pred_leaf_hi2 = 0;
  if ((! hsm.redundant) && (hsm.kbest == 0))
  { uint32_t top = min(hsm.leaf, hsm.K - hsm.leaf * pred0);
    base.multipredict(ec, hsm.root + hsm.leaf * pred0, top, hsm.pred_leaf, false);
    //cerr << "hsm.pred" << pred0 << " ="; for (size_t i=0; i<top; i++) cerr << ' ' << hsm.pred[i].scalar; cerr << endl;
    for (uint32_t i=1; i<top; i++)
      if (hsm.pred_leaf[i].scalar < hsm.pred_leaf[pred1].scalar)
        pred1 = i;
    prediction = pred0 * hsm.leaf + pred1 + 1;
    assert(prediction <= hsm.K);
    assert(prediction > 0);
  } else if ((! hsm.redundant) && (hsm.kbest > 0))
  {
    float pred1_val = FLT_MAX;
    for (size_t k=0; k<hsm.kbest; k++)
    { uint32_t predk = hsm.kbest_topk[k].second;
      float predk_v = hsm.kbest_topk[k].first;
      //cerr << hsm.kbest_topk[k].first << ": ";
      uint32_t top = min(hsm.leaf, hsm.K - hsm.leaf * predk);
      base.multipredict(ec, hsm.root + hsm.leaf * predk, top, hsm.pred_leaf, false);
      //cerr << "hsm.pred" << pred0 << " ="; for (size_t i=0; i<top; i++) cerr << ' ' << hsm.pred[i].scalar; cerr << endl;
      for (uint32_t i=0; i<top; i++)
      { //cerr << ' ' << hsm.pred_leaf[i].scalar;
        if (hsm.pred_leaf[i].scalar + predk_v < pred1_val)
        { pred1 = i;
          pred1_val = hsm.pred_leaf[i].scalar + predk_v;
          prediction = predk * hsm.leaf + pred1 + 1;
          //cerr << '*';
          //if (k > 0) cerr << "WOW!";
        }
      }
      //cerr << endl;
    }
    //cerr << endl;
    assert(prediction <= hsm.K);
    assert(prediction > 0);
  } else
  { // redundant representation needs a bit more work. except for
    // first L/2 classes, we need to make L predictions, starting
    // at position root + (L/2)*pred0
    //cerr << disp(pred0) << disp(hsm.leaf) << endl;
    if (pred0 != hsm.root-1)
    { base.multipredict(ec, hsm.root + hsm.leaf/2 * pred0, hsm.leaf, hsm.pred_leaf, false);
      pred_leaf_lo1 = hsm.leaf/2 * pred0;
      pred_leaf_hi1 = pred_leaf_lo1 + hsm.leaf;
      //cerr << " hsm.pred_leaf0=[" ; for (uint32_t i=0; i<hsm.leaf; i++) cerr << ' ' << hsm.pred_leaf[i].scalar; cerr << " ]" << endl;
      for (uint32_t i=1; i<hsm.leaf; i++)
        if (hsm.pred_leaf[i].scalar < hsm.pred_leaf[pred1].scalar)
          pred1 = i;
      prediction = hsm.leaf/2 * pred0 + pred1 + 1;
    } else
    { // when pred0=L-1, we need to predict the last few classes (<=L/2) and the first L/2 classes
      base.multipredict(ec, hsm.root + 0, hsm.leaf/2, hsm.pred_leaf, false);  // predict k=1,2,3,...,L/2
      uint32_t top = (hsm.K < hsm.leaf/2 * pred0) ? 0 : (hsm.K - hsm.leaf/2 * pred0);
      base.multipredict(ec, hsm.root + hsm.leaf/2 * pred0, top, hsm.pred_leaf + hsm.leaf/2, false); // predict the last classes at pred_leaf[L/2...]
      pred_leaf_lo1 = 0;
      pred_leaf_hi1 = hsm.leaf/2;
      pred_leaf_lo2 = hsm.leaf/2 * pred0;
      pred_leaf_hi2 = pred_leaf_lo2 + hsm.leaf/2;
      //cerr << disp(top) << disp(pred_leaf_lo1) << disp(pred_leaf_hi1) << disp(pred_leaf_lo2) << disp(pred_leaf_hi2) << endl;
      //cerr << " hsm.pred_leaf1=[" ; for (uint32_t i=0; i<hsm.leaf; i++) cerr << ' ' << hsm.pred_leaf[i].scalar; cerr << " ]" << endl;
      for (uint32_t i=1; i<hsm.leaf/2 + top; i++)
        if (hsm.pred_leaf[i].scalar < hsm.pred_leaf[pred1].scalar)
          pred1 = i;
      if (pred1 < hsm.leaf/2)
        prediction = pred1 + 1;
      else
        prediction = hsm.leaf/2 * pred0 + (pred1 - hsm.leaf/2) + 1;
    }
    assert(prediction <= hsm.K);
    assert(prediction > 0);
  }
  //cerr << disp(pred0) << disp(pred1) << disp(prediction) << endl;
  if (is_learn) {
    float min_cost = FLT_MAX, max_cost = -FLT_MAX;
    // at the bottom level, we update (all minima) \cup (current prediction)
    hsm.update_bottom.clear();  // which ids
    for (size_t i=0; i<hsm.root; i++)
      hsm.top_costs[i] = FLT_MAX;

    for (CS::wclass& wc : ld.costs)
    { min_cost = min(min_cost, wc.x);
      max_cost = max(max_cost, wc.x);
    }
    if (!hsm.redundant)
    { for (CS::wclass& wc : ld.costs)
      { size_t j = (wc.class_index-1) / hsm.leaf;
        if (wc.x <= min_cost)
          hsm.update_bottom.insert(j);
        hsm.top_costs[j] = min(hsm.top_costs[j], wc.x);
      }
    } else
    { // redundant
      for (CS::wclass& wc : ld.costs)
      { uint32_t j, j2;
        if (wc.class_index <= hsm.leaf/2)
        { j  = 0;
          j2 = hsm.root -1;
        } else
        { j  = (wc.class_index-1-hsm.leaf/2) / (hsm.leaf/2);
          j2 = j + 1;
        }
        if (wc.x <= min_cost)
        { hsm.update_bottom.insert(j);
          hsm.update_bottom.insert(j2);
        }
        hsm.top_costs[j ] = min(hsm.top_costs[j ], wc.x);
        hsm.top_costs[j2] = min(hsm.top_costs[j2], wc.x);
      }
    }
    //cerr << "hsm.update_bottom = "; for (auto x : hsm.update_bottom) cerr << x << ' '; cerr << endl;
    //cerr << "hsm.top_costs     = "; for (auto x : hsm.top_costs) cerr << x << ' '; cerr << endl;

    ec.l.simple = { 0.f, 1.f, 0.f };
    uint32_t new_pred0 = 0;
    float    new_pred0_v = FLT_MAX;
    for (size_t i=0; i<hsm.root; i++)
      if (hsm.top_costs[i] < FLT_MAX)
      { set_label(hsm, ec.l.simple, hsm.top_costs[i], min_cost, max_cost);
        ec.partial_prediction = hsm.pred_root[i].scalar;
        ec.pred.scalar = ec.partial_prediction;
        //cerr << "learn0(" << i << ", " << ec.l.simple.label << ")" << endl;
        base.update(ec, i); // TODO: update
        if (ec.partial_prediction < new_pred0_v)
        { new_pred0_v = ec.partial_prediction;
          new_pred0   = i;
        }
        if (hsm.kbest > 0)
        { hsm.kbest_topk[i].first = ec.partial_prediction; // hsm.pred_root[i].scalar;
          hsm.kbest_topk[i].second = i;
        }
      } else if (hsm.kbest > 0)
      { hsm.kbest_topk[i].first = FLT_MAX;
        hsm.kbest_topk[i].second = i;
      }
    if (hsm.filter)
      hsm.update_bottom.clear();
    if (hsm.kbest == 1)
      hsm.update_bottom.insert(new_pred0);
    else
    { std::sort(hsm.kbest_topk.begin(), hsm.kbest_topk.begin() + hsm.root);
      for (size_t k=0; k<hsm.kbest; k++)
        hsm.update_bottom.insert(hsm.kbest_topk[k].second);
    }
    
    for (CS::wclass& wc : ld.costs)
    { size_t j,j2;
      if (! hsm.redundant)
      { j  = (wc.class_index-1) / hsm.leaf;
        j2 = j;
      } else // redundant
      { if (wc.class_index <= hsm.leaf/2)
        { j  = 0;
          j2 = hsm.root-1;
        } else // normal
        { j  = (wc.class_index-1-hsm.leaf/2) / (hsm.leaf/2);
          j2 = j + 1;
        }
      }
      if ((hsm.update_bottom.find(j)  != hsm.update_bottom.end()) ||
          ((j != j2) && (hsm.update_bottom.find(j2) != hsm.update_bottom.end())))
      { set_label(hsm, ec.l.simple, wc.x, min_cost, max_cost);
        if ((! hsm.redundant) && (j == pred0))
        { ec.partial_prediction = hsm.pred_leaf[(wc.class_index-1) % hsm.leaf].scalar;
          ec.pred.scalar = ec.partial_prediction;
          base.update(ec, hsm.root + wc.class_index - 1);
          //cerr << "learn1(root + " << wc.class_index << " - 1, " << ec.l.simple.label << ")" << endl;
        }
        else if (hsm.redundant && (pred_leaf_lo1 <= wc.class_index-1) && (wc.class_index-1 < pred_leaf_hi1))
        { //cerr << disp(wc.class_index) << disp(pred_leaf_lo1) << disp(pred_leaf_hi1) << endl;
          ec.partial_prediction = hsm.pred_leaf[wc.class_index-1 - pred_leaf_lo1].scalar;
          ec.pred.scalar = ec.partial_prediction;
          base.update(ec, hsm.root + wc.class_index - 1);
          //cerr << "learn2(root + " << wc.class_index << " - 1, " << ec.l.simple.label << ")" << endl;
        }
        else if (hsm.redundant && (pred_leaf_lo2 <= wc.class_index-1) && (wc.class_index-1 < pred_leaf_hi2))
        { ec.partial_prediction = hsm.pred_leaf[wc.class_index-1 - pred_leaf_lo2 + hsm.leaf/2].scalar;
          ec.pred.scalar = ec.partial_prediction;
          base.update(ec, hsm.root + wc.class_index - 1);
          //cerr << "learn3(root + " << wc.class_index << " - 1, " << ec.l.simple.label << ")" << endl;
        } else
        { base.learn(ec, hsm.root + wc.class_index - 1);
          //cerr << "learn4(root + " << wc.class_index << " - 1, " << ec.l.simple.label << ")" << endl;
        }          
        //cerr << "learn1(hsm.root + " << wc.class_index << " - 1)" << endl;
      }
    }
  }

  ec.pred.multiclass = prediction;
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
  c.kbest_topk.delete_v();
}


base_learner* cshsm_setup(vw& all)
{ if (missing_option<size_t, true>(all, "cshsm", "One-against-all multiclass with <k> costs via hierarchical softmax"))
    return nullptr;

  cshsm& c = calloc_or_throw<cshsm>();

  c.K = (uint32_t)all.vm["cshsm"].as<size_t>();

  new_options(all, "CSHSM Options")
      ("classificationesque", "switch CSHSM into classification mode")
      ("hsm_branch", po::value<uint32_t>(&c.root)->default_value(2 * (int)ceilf(sqrt((float)c.K))), "branching factor at root")
      ("redundant", "use redundant representation (each class appears in two leaves instead of one)")
      ("filter_tree", "do optimistic filter-tree optimization (ie don't update root if leaf cannot succeed & vice versa)")
      ("kbest_tree", po::value<size_t>(), "do one-versus some between top K predicted leaves")
      ("initial_cost", po::value<float>(&c.initial)->default_value(0.), "set the initial prediction value (default 0.; 1. makes cshsm pessimistic)");
  add_options(all);

  c.redundant = all.vm.count("redundant") > 0;
  c.kbest = (all.vm.count("kbest_tree") == 0) ? 1 : all.vm["kbest_tree"].as<size_t>();
  c.filter = all.vm.count("filter_tree") > 0;
  if (c.redundant && c.filter)
    THROW("cannot do both --redundant and --filter yet" << endl);
  if ((c.kbest > 1) && (c.redundant))
    THROW("cannot do both --redundant and --kbest_tree" << endl);
  c.classificationesque = all.vm.count("classificationesque") > 0;
  c.leaf = (uint32_t)ceilf((float)c.K / (float)c.root);
  c.root = (uint32_t)ceilf((float)c.K / (float)c.leaf); // correct if overshot
  size_t ceil_K = c.root * (c.leaf+1);
  if (c.redundant)
    c.leaf *= 2;

  c.kbest = min(c.kbest, c.root);
  c.kbest_topk = v_init<pair<float,uint32_t>>();
  for (size_t k=0; k<c.K; k++)
    c.kbest_topk.push_back( make_pair(FLT_MAX, 0) );
  
  //cerr << disp(c.root) << disp(c.leaf) << disp (ceil_K) << endl;
  
  c.pred_root = calloc_or_throw<polyprediction>(c.root);
  c.pred_leaf = calloc_or_throw<polyprediction>(c.leaf);
  c.top_costs = v_init<float>();
  for (size_t i=0; i<c.root; i++)
    c.top_costs.push_back(FLT_MAX);
  
  learner<cshsm>& l = init_learner(&c,
                                   setup_base(all),
                                   predict_or_learn<true>,
                                   predict_or_learn<false>,
                                   (uint32_t)ceil_K);
  all.p->lp = CS::cs_label;
  l.set_finish_example(finish_example);
  l.set_finish(finish);
  base_learner* b = make_base(l);
  all.cost_sensitive = b;
  return b;
}

/*
  how can we have redundant representations?

  right now the tree looks like:

    [ [1 2 3 4] [5 6 7 8] [9 10 11 12] ... [997 998 999 1000] ]

  since these are roughly sorted by similarity, it seems something
  reasonable might be to have overlapping adjacent groups, ala:

    [ [1 2 3 4 5 6] [4 5 6 7 8 9] [7 8 9 10 11 12] [10 11 12 13 14 15]
      [13 14 15 16 17 18] ... [998 999 1000 1 2 3] ]

  if we let L = # of items in a single leaf (so L=6 above), then,
  class k gets assigned to:

    (k-1)/L and 1+(k-1)/L

  unless k<=(L/2) [i.e., k=1,2,3 above], then it is in

    0 and (K-1)/L


  how does filter tree work?

  suppose we have:

    [ A[1 2 3 4] B[5 6 7 8] C[9 10 11 12] ... [997 998 999 1000] ]

  and let's say 1:0 and 5:0 and 9:0.5 and everything else :1.

  "normal" updating would update A and B, meaning [1..8], as well as
  updating at the root A:0 B:0 C:0.5 and the rest :1. [9..12] would
  only get updated if C were predicted at the top level.

  at the root level, the ideal behavior would be to assign A:c(A)
  B:c(B), ..., where c(A) = cost of whatever prediction the [1..4]
  classifier made, etc. this is infeasible because we'd have to
  evaluate all [1..1000].

  the "normal" updating rule is essentially "super optimistic" in that
  it assumes that the lower-level subproblems can be solved
  perfectly. this means it will always prefer A,B over C, even though
  it might be that A,B are impossible (and might even suffer a loss of
  1) and C is possible.

  a slightly less optimistic version would be to consider the branch
  that is actually predicted at the root (not sure if before or after
  update) and and use the cost of the thing predicted under that
  (again, perhaps before or after update). for instance, if A was
  predicted at root, but 2 was selected under A, then the root update
  should be :1. for this, probably we should be optimistic at the leaf
  and do this after leaf update. for which branch to look at, it's
  probably better to do after update, but that's expensive because
  we'd have to repredict at leaf. but maybe we're doing that anyway if
  we're updating the leaves?

  okay, so general story is we first make a prediction at root and at
  corresponding leaf like normal.

  then, we take a collection of branches. this will include the
  minimum (optimistic) cost branch(es) as well as the predicted
  branch. we update all leaves under these branches as usual, but
  collect the (optimistic, post-update) winner at each branch. the
  cost of that winner becomes the cost of these branches. the cost of
  the remaining branches is the totally optimistic cost.

  
  
*/
