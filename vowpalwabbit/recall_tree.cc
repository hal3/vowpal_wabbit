/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved. Released under a BSD (revised)
license as described in the file LICENSE.node
*/
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>

#include "reductions.h"

using namespace std;
using namespace LEARNER;

namespace recall_tree {

class node_label_stats
{
public:

  uint32_t label;
  uint32_t label_count; // TODO: need floats for importance weighted examples

  bool operator==(node_label_stats v)
  { return (label == v.label);
  }

  bool operator>(node_label_stats v)
  { if(label > v.label) return true;
    return false;
  }

  bool operator<(node_label_stats v)
  { if(label < v.label) return true;
    return false;
  }

  node_label_stats(uint32_t l)
  { label = l;
    label_count = 0;
  }
};

typedef struct
{ //everyone has
  uint32_t parent;//the parent node
  v_array<node_label_stats> labelstats;//per-class state, in sorted by frequency descending
  float recall_lbest;

  bool internal;//internal or leaf
  uint32_t depth;

  //internal nodes have
  uint32_t base_router;//id of the base router
  uint32_t left;//left child
  uint32_t right;//right child
  uint32_t n;//total events at the node
  double sum_pred;

} node;

struct recall_tree
{ uint32_t k;

  v_array<node> nodes;

  size_t max_routers;
  size_t routers_used;
  size_t max_depth;
  float sanders;
  size_t max_candidates;
};

inline void init_leaf(node& n)
{ n.internal = false;
  n.base_router = 0;
  n.left = 0;
  n.right = 0;
  n.n = 0;
  n.sum_pred = 0.;
}

inline node init_node()
{ node node;

  node.parent = 0;
  node.depth = 0;
  node.labelstats = v_init<node_label_stats>();
  node.recall_lbest = 0;
  init_leaf(node);

  return node;
}

void init_tree(recall_tree& b, uint32_t root, uint32_t depth)
{
  if (depth < b.max_depth)
    {
      uint32_t left_child;
      uint32_t right_child;
      left_child = (uint32_t)b.nodes.size();
      b.nodes.push_back(init_node());
      right_child = (uint32_t)b.nodes.size();
      b.nodes.push_back(init_node());
      b.nodes[root].base_router = (uint32_t)b.routers_used++;

      b.nodes[root].internal = true;
      b.nodes[root].left = left_child;
      b.nodes[left_child].parent = root;
      b.nodes[left_child].depth = depth;
      b.nodes[root].right = right_child;
      b.nodes[right_child].parent = root;
      b.nodes[right_child].depth = depth;

      init_tree (b, left_child, depth+1);
      init_tree (b, right_child, depth+1);
    }
}

void init_tree(recall_tree& b)
{ b.nodes.push_back(init_node());
  init_tree (b, 0, 0);
  b.max_routers = b.nodes.size();
  assert (b.routers_used <= b.max_routers);
}

inline uint32_t descend(node& n, float prediction)
{ 
  float avg_pred = n.sum_pred / std::max (n.n, 1U);

//  std::cerr << " node = " << &n 
//            << " prediction = " << prediction << " avg_pred =  " << avg_pred 
//            << " sum_pred = " << n.sum_pred << " n.n = " << n.n << std::endl;

  if (prediction < avg_pred)
    return n.left;
  else
    return n.right;
}

uint32_t predict_from(recall_tree& b,  base_learner& base, example& ec, uint32_t cn, uint32_t depth)
{ MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  ec.l.simple = {FLT_MAX, 0.f, 0.f};
  while(b.nodes[cn].internal)
  { base.predict(ec, b.nodes[cn].base_router); // depth
    uint32_t newcn = descend(b.nodes[cn], ec.pred.scalar);
//    if (! (b.nodes[cn].recall_lbest >= b.nodes[newcn].recall_lbest))
//    if (depth == 0)
//    std::cerr << " depth = " << depth 
//              << " b.nodes[" << cn << "].recall_lbest = " << b.nodes[cn].recall_lbest
//              << " b.nodes[" << newcn << "].recall_lbest = " << b.nodes[newcn].recall_lbest
//              << " cond " << (b.nodes[cn].recall_lbest >= b.nodes[newcn].recall_lbest)
//              << std::endl;

    if (b.nodes[cn].recall_lbest >= b.nodes[newcn].recall_lbest)
      break;
    cn = newcn;
    depth ++;
  }
  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return cn;
}

float candidate_from (recall_tree& b,  base_learner& base, example& ec, uint32_t cn, uint32_t depth, uint32_t* mega = 0)
{
  const double epsilon = 1e-6;
  uint32_t leaf = predict_from (b, base, ec, cn, depth);
  if (mega) *mega=leaf;

  double min_cand_label_count = 
    b.nodes[leaf].labelstats.size () < b.max_candidates 
      ? 0.0 : b.nodes[leaf].labelstats[b.max_candidates].label_count;

  double true_label_count = 0;

  size_t index;
  for (index = 0; index < b.nodes[leaf].labelstats.size (); ++index)
    {
      node_label_stats* ls = b.nodes[leaf].labelstats.begin + index;

      assert (ls + 1 == b.nodes[leaf].labelstats.end ||
              ls[0].label_count >= ls[1].label_count || 
              (std::cerr << "ls[0].label_count = " << ls[0].label_count 
                         << " ls[1].label_count = " << ls[1].label_count 
                         << std::endl, 0));


      if (ls->label == ec.l.multi.label)
        {
          true_label_count = ls->label_count;

          if (index < b.max_candidates)
            {
              // NB: no epsilon on purpose 

              assert (min_cand_label_count <= true_label_count);
              return min_cand_label_count - true_label_count;
            }
          else
            {
              assert (min_cand_label_count >= true_label_count);
              return std::max (min_cand_label_count - true_label_count,
                               epsilon);
            }
        }
    }

  return std::max (min_cand_label_count, epsilon);
}

void predict(recall_tree& b,  base_learner& base, example& ec)
{ 
  uint32_t leaf; 
  ec.pred.multiclass = candidate_from (b, base, ec, 0, 0, &leaf) > 0.f ? 1 + ec.l.multi.label : ec.l.multi.label;
  ec.num_features = leaf;
}

float train_node(recall_tree& b, base_learner& base, example& ec, uint32_t& current, uint32_t depth)
{ 
  float candidate_left = candidate_from (b, base, ec, b.nodes[current].left, 1+b.nodes[current].depth);
  float candidate_right = candidate_from (b, base, ec, b.nodes[current].right, 1+b.nodes[current].depth);

  MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  ec.l.simple = { candidate_left < candidate_right ? -1.f : 1.f, 
                  std::min (mc.weight, 
                            std::abs (candidate_left - candidate_right)),
                  0. };

  base.learn(ec, b.nodes[current].base_router);	// depth

  b.nodes[current].n++; // TODO: importance weight from example
  b.nodes[current].sum_pred += ec.pred.scalar;

  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return ec.partial_prediction;
}

void insert_example_at_node (recall_tree& b, uint32_t cn, example& ec)
{
  node_label_stats* ls;

  for (ls = b.nodes[cn].labelstats.begin;
       ls != b.nodes[cn].labelstats.end && ls->label != ec.l.multi.label;
       ++ls);

  if (ls == b.nodes[cn].labelstats.end) {
    node_label_stats newls (ec.l.multi.label); newls.label_count++;
    b.nodes[cn].labelstats.push_back (newls);
  }
  else {
    ls->label_count++;

    while (ls != b.nodes[cn].labelstats.begin &&
          ls[-1].label_count < ls[0].label_count) {
      std::swap (ls[-1], ls[0]);
      assert (ls[-1].label_count >= ls[0].label_count);
      --ls;
    }
  }

  b.nodes[cn].n++; // TODO: importance weight from example

  uint32_t mass_at_k = 0;

  for (ls = b.nodes[cn].labelstats.begin;
       ls != b.nodes[cn].labelstats.end && ls < b.nodes[cn].labelstats.begin + b.max_candidates; 
       ++ls)
    {
      mass_at_k += ls->label_count;
      assert (ls + 1 == b.nodes[cn].labelstats.end ||
              ls[0].label_count >= ls[1].label_count);
    }

  float f = (float) mass_at_k / (float) b.nodes[cn].n;
  float stdf = sqrt (f * (1. - f) / (float) b.nodes[cn].n);
  float diamf = 15. / (sqrt (18.) * (float) b.nodes[cn].n);

  b.nodes[cn].recall_lbest = std::max (0.f, f - b.sanders * (stdf + diamf));
}

void learn(recall_tree& b, base_learner& base, example& ec)
{ 
  predict(b,base,ec);

  if(ec.l.multi.label != (uint32_t)-1)	//if training the tree
  { 
    uint32_t cn = 0;
    uint32_t depth = 0;

    while(b.nodes[cn].internal)
    { 
      insert_example_at_node (b, cn, ec);
      cn = descend(b.nodes[cn], train_node (b, base, ec, cn, depth));
      depth++;
    }
    insert_example_at_node (b, cn, ec);
  }
}

void finish(recall_tree& b)
{ for (size_t i = 0; i < b.nodes.size(); i++)
    b.nodes[i].labelstats.delete_v();
  b.nodes.delete_v();
}

base_learner* recall_tree_setup(vw& all)	//learner setup
{ if (missing_option<size_t, true>(all, "recall_tree", "Use online tree for multiclass"))
    return nullptr;
  new_options(all, "Logarithmic Time Multiclass options")
  ("max_candidates", po::value<uint32_t>(), "maximum number of labels per leaf in the tree, default k/10 something")
  ("max_depth", po::value<uint32_t>(), "maximum number of labels per leaf in the tree, default log(k)/log(2) something")
  ("bern_hyper", po::value<float>(), "bound hyperparameter, bigger is more pessisimistic, default 1");
  add_options(all);

  po::variables_map& vm = all.vm;

  recall_tree& data = calloc_or_throw<recall_tree>();
  data.k = (uint32_t)vm["recall_tree"].as<size_t>();
  data.max_candidates = vm.count("max_candidates") > 0 ? vm["max_candidates"].as<uint32_t>() : ceil (data.k / 10.0);
  data.sanders = vm.count("bern_hyper") > 0 ? vm["bern_hyper"].as<float>() : 1.;
  data.max_depth = vm.count("max_depth") > 0 ? vm["max_depth"].as<uint32_t>() : (int) ceil (log (data.k) / log (2.0));
  init_tree(data);
  std::cerr << " data.max_routers = " << data.max_routers 
            << " data.max_depth = " << data.max_depth 
            << " data.max_candidates = " << data.max_candidates
            << " data.sanders (bern_hyper) = " << data.sanders
            << std::endl;

  learner<recall_tree>& l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.max_routers);
  l.set_finish(finish);

  return make_base(l);
}

}
