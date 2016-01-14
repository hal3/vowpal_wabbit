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
  float recall_estimate;

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
  float recall_target;
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
  node.recall_estimate = 0;
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
    cn = descend(b.nodes[cn], ec.pred.scalar);
    depth ++;
  }
  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return cn;
}

float recall_from (recall_tree& b,  base_learner& base, example& ec, uint32_t cn, uint32_t depth, float optimistic, uint32_t* mega = 0)
{
  uint32_t leaf = predict_from (b, base, ec, cn, depth);
  assert (leaf > 0);
  if (mega) *mega=leaf;

  node_label_stats* ls;
  for (ls = b.nodes[leaf].labelstats.begin; 
       ls != b.nodes[leaf].labelstats.end && ls < b.nodes[leaf].labelstats.begin + b.max_candidates; 
       ++ls)
    {
//      std::cerr << "leaf = " << leaf 
//                << " ls[" << (ls - b.nodes[leaf].labelstats.begin) << "].label = " << ls->label 
//                << "( ls[" << (ls - b.nodes[leaf].labelstats.begin) << "].label_count = " << ls->label_count << ")"
//                << " ?= " << ec.l.multi.label << std::endl;

      assert (ls + 1 == b.nodes[leaf].labelstats.end ||
              ls[0].label_count >= ls[1].label_count || 
              (std::cerr << "ls[0].label_count = " << ls[0].label_count 
                         << " ls[1].label_count = " << ls[1].label_count 
                         << std::endl, 0));


      if (ls->label == ec.l.multi.label)
        return 1.0;
    }

  //std::cerr << "ls (" << ls << ") ?= b.nodes[leaf].labelstats.end (" << b.nodes[leaf].labelstats.end << ")" << std::endl;
  return b.nodes[leaf].labelstats.size () < b.max_candidates ? optimistic : 0.0;
}

void predict(recall_tree& b,  base_learner& base, example& ec)
{ 
  uint32_t leaf; 
  //ec.pred.multiclass = recall_from (b, base, ec, 0, 0, 0.0, &leaf) ? leaf : leaf ; 
  ec.pred.multiclass = recall_from (b, base, ec, 0, 0, 0.0, &leaf) ? ec.l.multi.label : (1 + ec.l.multi.label);
  ec.num_features = leaf;
  assert(leaf>0);
}

float train_node(recall_tree& b, base_learner& base, example& ec, uint32_t& current, uint32_t depth)
{ 
  MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  assert (b.nodes[current].internal);

  float recall_left = recall_from (b, base, ec, b.nodes[current].left, 1+b.nodes[current].depth, 1.0);
  float recall_right = recall_from (b, base, ec, b.nodes[current].right, 1+b.nodes[current].depth, 1.0);

//  ec.l.simple = { avg_pred > 0 ? -1.f : 1.f, std::abs (avg_pred), 0. };
//  base.learn(ec, b.nodes[current].base_router);	// depth

  //std::cerr << ( recall_left == recall_right  ? (recall_left < 0.5 ? "0" : "1") : ( recall_left > recall_right ? "-" : "+" ) );

  ec.l.simple = { recall_left > recall_right ? -1.f : 1.f, std::abs (recall_left - recall_right), 0. };

  //std::cerr << "ec.l.simple.label = " << ec.l.simple.label << " recall_left = " << recall_left << " recall_right = " << recall_right << std::endl;
  
  base.learn(ec, b.nodes[current].base_router);	// depth

  b.nodes[current].n++; // TODO: importance weight from example
  b.nodes[current].sum_pred += ec.pred.scalar;

  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return ec.partial_prediction;
}

void update_recall_estimate (recall_tree& b, uint32_t cn, example& ec)
{
  assert (! b.nodes[cn].internal);

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

    // TODO: this can be (much?) faster
    
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

  b.nodes[cn].recall_estimate = (float) mass_at_k / (float) b.nodes[cn].n;
}

void learn(recall_tree& b, base_learner& base, example& ec)
{ //    verify_min_dfs(b, b.nodes[0]);
  predict(b,base,ec);

  if(ec.l.multi.label != (uint32_t)-1)	//if training the tree
  { 
    uint32_t cn = 0;
    uint32_t depth = 0;

    while(b.nodes[cn].internal)
    { 
      cn = descend(b.nodes[cn], train_node (b, base, ec, cn, depth));
      depth++;
    }

    update_recall_estimate (b, cn, ec);
  }
}

//void save_node_stats(recall_tree& d)
//{ FILE *fp;
//  uint32_t i, j;
//  uint32_t total;
//  recall_tree* b = &d;
//
//  fp = fopen("atxm_debug.csv", "wt");
//
//  for(i = 0; i < b->nodes.size(); i++)
//  { fprintf(fp, "Node: %4d, Internal: %1d, Eh: %7.4f, n: %6d, \n", (int) i, (int) b->nodes[i].internal, b->nodes[i].Eh / b->nodes[i].n, b->nodes[i].n);
//
//    fprintf(fp, "Label:, ");
//    for(j = 0; j < b->nodes[i].labelstats.size(); j++)
//    { fprintf(fp, "%6d,", (int) b->nodes[i].labelstats[j].label);
//    }
//    fprintf(fp, "\n");
//
//    fprintf(fp, "Ehk:, ");
//    for(j = 0; j < b->nodes[i].labelstats.size(); j++)
//    { fprintf(fp, "%7.4f,", b->nodes[i].labelstats[j].Ehk / b->nodes[i].labelstats[j].nk);
//    }
//    fprintf(fp, "\n");
//
//    total = 0;
//
//    fprintf(fp, "nk:, ");
//    for(j = 0; j < b->nodes[i].labelstats.size(); j++)
//    { fprintf(fp, "%6d,", (int) b->nodes[i].labelstats[j].nk);
//      total += b->nodes[i].labelstats[j].nk;
//    }
//    fprintf(fp, "\n");
//
//    fprintf(fp, "max(lab:cnt:tot):, %3d,%6d,%7d,\n", (int) b->nodes[i].max_count_label, (int) b->nodes[i].max_count, (int) total);
//    fprintf(fp, "left: %4d, right: %4d", (int) b->nodes[i].left, (int) b->nodes[i].right);
//    fprintf(fp, "\n\n");
//  }
//
//  fclose(fp);
//}

void finish(recall_tree& b)
{ //save_node_stats(b);
  for (size_t i = 0; i < b.nodes.size(); i++)
    b.nodes[i].labelstats.delete_v();
  b.nodes.delete_v();
}

// void save_load_tree(recall_tree& b, io_buf& model_file, bool read, bool text)
// { if (model_file.files.size() > 0)
//     { stringstream msg;
//       msg << "k = " << b.k;
//       bin_text_read_write_fixed(model_file,(char*)&b.max_routers, sizeof(b.k), "", read, msg, text);
// 
//       msg << "nodes = " << b.nodes.size() << " ";
//       uint32_t temp = (uint32_t)b.nodes.size();
//       bin_text_read_write_fixed(model_file,(char*)&temp, sizeof(temp), "", read, msg, text);
//     if (read)
//       for (uint32_t j = 1; j < temp; j++)
//         b.nodes.push_back(init_node());
// 
//     msg << "max routers = " << b.max_routers << " ";
//     bin_text_read_write_fixed(model_file,(char*)&b.max_routers, sizeof(b.max_routers), "", read, msg, text);
// 
//     msg << "routers_used = " << b.routers_used << " ";
//     bin_text_read_write_fixed(model_file,(char*)&b.routers_used, sizeof(b.routers_used), "", read, msg, text);
// 
//     msg << "progress = " << b.progress << " ";
//     bin_text_read_write_fixed(model_file,(char*)&b.progress, sizeof(b.progress), "", read, msg, text);
// 
//     msg << "swap_resist = " << b.swap_resist << "\n";
//     bin_text_read_write_fixed(model_file,(char*)&b.swap_resist, sizeof(b.swap_resist), "", read, msg, text);
// 
//     for (size_t j = 0; j < b.nodes.size(); j++)
//     { //Need to read or write nodes.
//       node& n = b.nodes[j];
// 
//       msg << " parent = " << n.parent;
//       bin_text_read_write_fixed(model_file,(char*)&n.parent, sizeof(n.parent), "", read, msg, text);
// 
//       uint32_t temp = (uint32_t)n.labelstats.size();
// 
//       msg << " labelstats = " << temp;
//       bin_text_read_write_fixed(model_file,(char*)&temp, sizeof(temp), "", read, msg, text);
//       if (read)
//         for (uint32_t k = 0; k < temp; k++)
//           n.labelstats.push_back(node_label_stats(1));
// 
//       msg << " min_count = " << n.min_count;
//       bin_text_read_write_fixed(model_file,(char*)&n.min_count, sizeof(n.min_count), "", read, msg, text);
// 
//       msg << " internal = " << n.internal;
//       bin_text_read_write_fixed(model_file,(char*)&n.internal, sizeof(n.internal), "", read, msg, text);
// 
//       if (n.internal)
//         { msg << " base_router = " << n.base_router;
//         bin_text_read_write_fixed(model_file,(char*)&n.base_router, sizeof(n.base_router), "", read, msg, text);
// 
//         msg << " left = " << n.left;
//         bin_text_read_write_fixed(model_file,(char*)&n.left, sizeof(n.left), "", read, msg, text);
// 
//         msg << " right = " << n.right;
//         bin_text_read_write_fixed(model_file,(char*)&n.right, sizeof(n.right), "", read, msg, text);
// 
//         msg << " norm_Eh = " << n.norm_Eh;
//         bin_text_read_write_fixed(model_file,(char*)&n.norm_Eh, sizeof(n.norm_Eh), "", read, msg, text);
// 
//         msg << " Eh = " << n.Eh;
//         bin_text_read_write_fixed(model_file,(char*)&n.Eh, sizeof(n.Eh), "", read, msg, text);
// 
//         msg << " n = "<< n.n << "\n";
//         bin_text_read_write_fixed(model_file,(char*)&n.n, sizeof(n.n), "", read, msg, text);
//       }
//       else
//         { msg << " max_count = " << n.max_count;
//           bin_text_read_write_fixed(model_file,(char*)&n.max_count, sizeof(n.max_count), "", read, msg, text);
//           msg << " max_count_label = "<< n.max_count_label <<"\n";
//           bin_text_read_write_fixed(model_file,(char*)&n.max_count_label, sizeof(n.max_count_label), "", read, msg, text);
//         }
// 
//       for (size_t k = 0; k < n.labelstats.size(); k++)
//       { node_label_stats& p = n.labelstats[k];
// 
//         msg << "  Ehk = " << p.Ehk;
//         bin_text_read_write_fixed(model_file,(char*)&p.Ehk, sizeof(p.Ehk), "", read, msg, text);
// 
//         msg << " norm_Ehk = " << p.norm_Ehk;
//         bin_text_read_write_fixed(model_file,(char*)&p.norm_Ehk, sizeof(p.norm_Ehk), "", read, msg, text);
// 
//         msg << " nk = " << p.nk;
//         bin_text_read_write_fixed(model_file,(char*)&p.nk, sizeof(p.nk), "", read, msg, text);
// 
//         msg << " label = " << p.label;
//         bin_text_read_write_fixed(model_file,(char*)&p.label, sizeof(p.label), "", read, msg, text);
// 
//         msg << " label_count = "<< p.label_count << "\n";
//         bin_text_read_write_fixed(model_file,(char*)&p.label_count, sizeof(p.label_count), "", read, msg, text);
//       }
//     }
//   }
// }

base_learner* recall_tree_setup(vw& all)	//learner setup
{ if (missing_option<size_t, true>(all, "recall_tree", "Use online tree for multiclass"))
    return nullptr;
  new_options(all, "Logarithmic Time Multiclass options")
  ("max_candidates", po::value<uint32_t>(), "maximum number of labels per leaf in the tree, default k/10 something")
  ("max_depth", po::value<uint32_t>(), "maximum number of labels per leaf in the tree, default log(k)/log(2) something");
  add_options(all);

  po::variables_map& vm = all.vm;

  recall_tree& data = calloc_or_throw<recall_tree>();
  data.k = (uint32_t)vm["recall_tree"].as<size_t>();
  data.recall_target = 1.0;
  data.max_candidates = vm.count("max_candidates") > 0 ? vm["max_candidates"].as<uint32_t>() : ceil (data.k / 10.0);
 
  data.max_depth = vm.count("max_depth") > 0 ? vm["max_depth"].as<uint32_t>() : (int) ceil (log (data.k) / log (2.0));
  init_tree(data);
  std::cerr << " data.max_routers = " << data.max_routers 
            << " data.max_depth = " << data.max_depth 
            << " data.max_candidates = " << data.max_candidates
            << std::endl;

  learner<recall_tree>& l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.max_routers);
  //l.set_save_load(save_load_tree);
  l.set_finish(finish);

  return make_base(l);
}

}
