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

    

namespace oas {

template<class T> struct pqueue {
  public:
  v_array<v_array<pair<T,float>>> A;
  uint32_t level;
  uint32_t index; // we're looking at A[level][index]
  float live_mass;

  void insert(T data, float cost) {
    uint32_t l = (uint32_t)floor(cost);
    if (l >= A.size()) l = A.size()-1;
    A[l].push_back( make_pair(data,cost) );
    live_mass += exp(-cost);
  }

  pair<T,float>* next() {
    if (level >= A.size()) return nullptr;
    if (index >= A[level].size()) {
      A[level].erase();
      level++;
      index=0;
      return next();
    }
    live_mass -= exp(-A[level][index].second);
    return &A[level][index++];
  }

  void reset() {
    for (size_t l=level; l<A.size(); l++)
      A[l].erase();
    index = 0;
    level = 0;
    live_mass = 0.;
  }

  pqueue(uint32_t max_levels) {
    for (uint32_t i=0; i<max_levels; i++)
      A.push_back( v_init<pair<T,float>>() );
    level = 0;
    index = 0;
    live_mass = 0.;
  }
};
  
class node_pred
{
public:

  double Ehk;
  float norm_Ehk;
  double  nk;
  uint32_t label;
  double   label_count;

  bool operator==(node_pred v)
  { return (label == v.label);
  }

  bool operator>(node_pred v)
  { if(label > v.label) return true;
    return false;
  }

  bool operator<(node_pred v)
  { if(label < v.label) return true;
    return false;
  }

  node_pred(uint32_t l)
  { label = l;
    Ehk = 0.f;
    norm_Ehk = 0;
    nk = 0.;
    label_count = 0.;
  }
};

typedef struct
{ //everyone has
  uint32_t parent;//the parent node
  v_array<node_pred> preds;//per-class state
  double min_count;//the number of examples reaching this node (if it's a leaf) or the minimum reaching any grandchild.

  bool internal;//internal or leaf

  //internal nodes have
  uint32_t base_predictor;//id of the base predictor
  uint32_t left;//left child
  uint32_t right;//right child
  float norm_Eh;//the average margin at the node
  double Eh;//total margin at the node
  double n;//total events at the node

  //leaf has
  double max_count;//the number of samples of the most common label
  uint32_t max_count_label;//the most common label
} node;

enum OASTrainMethod { TRAIN_SINGLE_PATH, TRAIN_HIT_NODES };


struct treenode {
  uint32_t value;
  bool is_class; // if false, value is a node in the tree; if true, value is the label
  treenode(uint32_t value, bool is_class) : value(value), is_class(is_class) {}
};

struct oas
{ uint32_t k; // size of tree (# of nodes)
  uint32_t num_classes; // number of classes (used for oas)

  v_array<node> nodes;

  size_t max_predictors;
  size_t predictors_used;

  bool evaluate_recall;
  uint32_t swap_resist;
  float log_MaxMassToConsider;
  uint32_t MaxDequeues;
  bool predict_by_sum;
  bool dequeue_leaves;
  
  uint32_t nbofswaps;

  OASTrainMethod train_method;
  pqueue<treenode> Q;
  float* class_scores;
  v_array<uint32_t> labelset;
  uint32_t num_repeats;
  uint32_t num_repeats_wrong;
  uint32_t num_classes_popped;
  float repeat_mass;
  float class_mass;
};

uint32_t treenode_hash(treenode& node) { return quadratic_constant * (node.value + cubic_constant * node.is_class); }

inline void init_leaf(node& n)
{ n.internal = false;
  n.preds.erase();
  n.base_predictor = 0;
  n.norm_Eh = 0;
  n.Eh = 0;
  n.n = 0.;
  n.max_count = 0.;
  n.max_count_label = 1;
  n.left = 0;
  n.right = 0;
}

inline node init_node()
{ node node;

  node.parent = 0;
  node.min_count = 0.;
  node.preds = v_init<node_pred>();
  init_leaf(node);

  return node;
}

void init_tree(oas& d)
{ d.nodes.push_back(init_node());
  d.nbofswaps = 0;
}

inline float min_left_right(oas& b, node& n)
{ return fmin(b.nodes[n.left].min_count, b.nodes[n.right].min_count);
}

inline uint32_t find_switch_node(oas& b)
{ uint32_t node = 0;
  while(b.nodes[node].internal)
    if(b.nodes[b.nodes[node].left].min_count
        < b.nodes[b.nodes[node].right].min_count)
      node = b.nodes[node].left;
    else
      node = b.nodes[node].right;
  return node;
}

inline void update_min_count(oas& b, uint32_t node)
{ //Constant time min count update.
  while(node != 0)
  { uint32_t prev = node;
    node = b.nodes[node].parent;

    if (b.nodes[node].min_count == b.nodes[prev].min_count)
      break;
    else
      b.nodes[node].min_count = min_left_right(b,b.nodes[node]);
  }
}

void display_tree_dfs(oas& b, node node, uint32_t depth)
{ for (uint32_t i = 0; i < depth; i++)
    cout << "\t";
  cout << node.min_count << " " << node.left
       << " " << node.right;
  cout << " label = " << node.max_count_label << " labels = ";
  for (size_t i = 0; i < node.preds.size(); i++)
    cout << node.preds[i].label << ":" << node.preds[i].label_count << "\t";
  cout << endl;

  if (node.internal)
  { cout << "Left";
    display_tree_dfs(b, b.nodes[node.left], depth+1);

    cout << "Right";
    display_tree_dfs(b, b.nodes[node.right], depth+1);
  }
}

bool children(oas& b, uint32_t& current, uint32_t& class_index, uint32_t label, float imp_weight)
{ class_index = (uint32_t)b.nodes[current].preds.unique_add_sorted(node_pred(label));
  b.nodes[current].preds[class_index].label_count+=imp_weight;

  if(b.nodes[current].preds[class_index].label_count > b.nodes[current].max_count)
  { b.nodes[current].max_count = b.nodes[current].preds[class_index].label_count;
    b.nodes[current].max_count_label = b.nodes[current].preds[class_index].label;
  }

  if (b.nodes[current].internal)
    return true;
  else if( b.nodes[current].preds.size() > 1
           && (b.predictors_used < b.max_predictors
               || b.nodes[current].min_count - b.nodes[current].max_count > b.swap_resist*(b.nodes[0].min_count + 1.)))
  { //need children and we can make them.
    uint32_t left_child;
    uint32_t right_child;
    if (b.predictors_used < b.max_predictors)
    { left_child = (uint32_t)b.nodes.size();
      b.nodes.push_back(init_node());
      right_child = (uint32_t)b.nodes.size();
      b.nodes.push_back(init_node());
      b.nodes[current].base_predictor = (uint32_t)b.predictors_used++;
    }
    else
    { uint32_t swap_child = find_switch_node(b);
      uint32_t swap_parent = b.nodes[swap_child].parent;
      uint32_t swap_grandparent = b.nodes[swap_parent].parent;
      if (b.nodes[swap_child].min_count != b.nodes[0].min_count)
        cout << "glargh " << b.nodes[swap_child].min_count << " != " << b.nodes[0].min_count << endl;
      b.nbofswaps++;

      uint32_t nonswap_child;
      if(swap_child == b.nodes[swap_parent].right)
        nonswap_child = b.nodes[swap_parent].left;
      else
        nonswap_child = b.nodes[swap_parent].right;

      if(swap_parent == b.nodes[swap_grandparent].left)
        b.nodes[swap_grandparent].left = nonswap_child;
      else
        b.nodes[swap_grandparent].right = nonswap_child;
      b.nodes[nonswap_child].parent = swap_grandparent;
      update_min_count(b, nonswap_child);

      init_leaf(b.nodes[swap_child]);
      left_child = swap_child;
      b.nodes[current].base_predictor = b.nodes[swap_parent].base_predictor;
      init_leaf(b.nodes[swap_parent]);
      right_child = swap_parent;
    }
    b.nodes[current].left = left_child;
    b.nodes[left_child].parent = current;
    b.nodes[current].right = right_child;
    b.nodes[right_child].parent = current;

    b.nodes[left_child].min_count = b.nodes[current].min_count/2.0;
    b.nodes[right_child].min_count = b.nodes[current].min_count - b.nodes[left_child].min_count;
    update_min_count(b, left_child);

    b.nodes[left_child].max_count_label = b.nodes[current].max_count_label;
    b.nodes[right_child].max_count_label = b.nodes[current].max_count_label;

    b.nodes[current].internal = true;
  }
  return b.nodes[current].internal;
}

void train_node(oas& b, base_learner& base, example& ec, uint32_t& current, uint32_t& class_index, float imp_weight)
{ ec.l.simple.label = (b.nodes[current].norm_Eh > b.nodes[current].preds[class_index].norm_Ehk) ? -1.f : 1.f;
  ec.l.simple.weight = imp_weight;

  base.learn(ec, b.nodes[current].base_predictor);	// depth

  ec.l.simple.label = FLT_MAX;
  base.predict(ec, b.nodes[current].base_predictor); // depth

  b.nodes[current].Eh += imp_weight * ec.partial_prediction;
  b.nodes[current].preds[class_index].Ehk += imp_weight * ec.partial_prediction;
  b.nodes[current].n += imp_weight;
  b.nodes[current].preds[class_index].nk += imp_weight;

  b.nodes[current].norm_Eh = (float)b.nodes[current].Eh / b.nodes[current].n;
  b.nodes[current].preds[class_index].norm_Ehk = (float)b.nodes[current].preds[class_index].Ehk / b.nodes[current].preds[class_index].nk;
}

void verify_min_dfs(oas& b, node node)
{ if (node.internal)
  { if (node.min_count != min_left_right(b, node))
    { cout << "badness! " << endl;
      display_tree_dfs(b, b.nodes[0], 0);
    }
    verify_min_dfs(b, b.nodes[node.left]);
    verify_min_dfs(b, b.nodes[node.right]);
  }
}

double sum_count_dfs(oas& b, node node)
{ if (node.internal)
    return sum_count_dfs(b, b.nodes[node.left]) + sum_count_dfs(b, b.nodes[node.right]);
  else
    return node.min_count;
}

inline uint32_t descend(node& n, float prediction)
{ if (prediction < 0.5)
    return n.left;
  else
    return n.right;
}

void predict_noucs(oas& b,  base_learner& base, example& ec)
{ MULTICLASS::label_t mc = ec.l.multi;

  ec.l.simple = {FLT_MAX, 0.f, 0.f};
  uint32_t cn = 0;
  uint32_t depth = 0;
  while(b.nodes[cn].internal)
  { base.predict(ec, b.nodes[cn].base_predictor); // depth
    cn = descend(b.nodes[cn], ec.pred.scalar);
    depth ++;
  }
  ec.pred.multiclass = b.nodes[cn].max_count_label;
  ec.l.multi = mc;
}

struct weighted_node {
  uint32_t node;
  float weight;
  weighted_node(uint32_t _node) : node(_node), weight(1.) {}
  weighted_node(uint32_t _node, float _weight) : node(_node), weight(_weight) {}
};

void predict_ucs(oas& b, base_learner& base, example& ec, v_array<weighted_node>*nodeset) // fill in labelset with the labels considered (if not nullptr), and nodeset with nodes hit (if not nullptr)
{
  MULTICLASS::label_t mc = ec.l.multi;
  ec.l.simple = {FLT_MAX, 0.f, 0.f};

  float total_mass = 0.;
 
  uint32_t pred = 1;
  float predScore = -FLT_MAX;
  uint32_t num_labels_considered = 0;
  
  b.labelset.erase();

  b.Q.reset();
  b.Q.insert(treenode(0,false), 0.f); // root node
  
  //uint32_t num_classes_popped_old = b.num_classes_popped;
  pair<treenode,float>* elem;
  while ((elem = b.Q.next()) != nullptr) {
    treenode& node = elem->first;
    float elem_cost = elem->second;
    //cerr << "popped is_class=" << node.is_class << " value=" << node.value << " cost=" << elem_cost << endl;
    if (node.is_class) {
      if (b.class_scores[node.value-1] <= 0.) { // if we haven't seen this label yet, we have to do work        
        b.labelset.push_back(node.value);

        if (b.evaluate_recall) {
          if (node.value == mc.label)
            pred = node.value;
        } else if (! b.predict_by_sum) {
          base.predict(ec, b.k + node.value-1);
          if (ec.partial_prediction > predScore) {
            predScore = ec.partial_prediction;
            pred = node.value;
          }
        }
      } else {
        b.num_repeats++;
        if (node.value != mc.label)
          b.num_repeats_wrong++;
        b.repeat_mass += exp(-elem_cost);
      }
      b.class_scores[node.value-1] += exp(-elem_cost);
      b.num_classes_popped++;
      b.class_mass += exp(-elem_cost);
      
      total_mass += exp(-elem_cost);
      num_labels_considered ++;
      if (log(total_mass) >= b.log_MaxMassToConsider || num_labels_considered >= b.MaxDequeues)
        break;
    } else { // node is an internal node
      uint32_t cn = node.value;
      if (nodeset != nullptr)
        nodeset->push_back(weighted_node(cn, exp(-elem_cost)));
      
      if (b.nodes[cn].internal) {
        base.predict(ec, b.nodes[cn].base_predictor);
        float pR = ec.pred.scalar; // (ec.pred.scalar + 1.) / 2.;
        //if (pR < 0.) pR = 0.; else if (pR > 1.) pR = 1.;
        //cerr << "pR=" << pR << endl;

        if (1. - pR > 0.) {
          treenode nodeL = { b.nodes[cn].left, false };
          //cerr << "push L is_class=0 value=" << b.nodes[cn].left << " cost=" << elem_cost - log(1.-pR) << endl;
          b.Q.insert(nodeL, elem_cost - log(1.-pR));
        }

        if (pR > 0.) {
          treenode nodeR = {b.nodes[cn].right, false };
          //cerr << "push R is_class=0 value=" << b.nodes[cn].right << " cost=" << elem_cost - log(pR) << endl;
          b.Q.insert(nodeR, elem_cost - log(pR));
        }
      } else { // leaf!
        double sum_count = 0.;
        for (node_pred*node=b.nodes[cn].preds.begin; node != b.nodes[cn].preds.end; ++node)
          if (node->label_count > 0)
            sum_count += node->label_count;
        if (sum_count == 0) {
          b.class_scores[0] += exp(-elem_cost);
          b.labelset.push_back(1);
          total_mass += exp(-elem_cost);
        } else if (sum_count > 0) {
          double log_sum_count = log(sum_count);
          for (node_pred*node=b.nodes[cn].preds.begin; node != b.nodes[cn].preds.end; ++node)
            if (node->label_count > 0) {
              if (b.class_scores[node->label - 1] > 0.) { // we've already seen this, just add to scores rather than reusing the queue
                b.class_scores[node->label - 1] += exp(-elem_cost) * node->label_count / sum_count;
                total_mass += exp(-elem_cost) * node->label_count / sum_count;
                // TODO: udpate mass
              } else {
                if (b.dequeue_leaves) {
                  b.labelset.push_back(node->label);
                  base.predict(ec, b.k + node->label-1);
                  if (ec.partial_prediction > predScore) {
                    predScore = ec.partial_prediction;
                    pred = node->label;
                  }
                } else {
                  treenode nodeL = { node->label, true };
                  double logp = log(node->label_count) - log_sum_count;
                  //cerr << "push C is_class=1 sum_count=" << exp(log_sum_count) << " label_count=" << node->label_count << " value=" << node->label << " cost=" << elem_cost - logp << endl;
                  b.Q.insert(nodeL, elem_cost - logp);
                }
              }
            }
        }
      }
    }
    assert(fabs(total_mass + b.Q.live_mass - 1.) < 1e-5);
  }
  //cerr << "total_mass " << (total_mass) << endl;
  assert(fabs(total_mass - 1.) < 1e-5);
  
  if (b.predict_by_sum) { // TODO use labelset to make this faster
    pred = 1;
    for (size_t i=2; i<=b.num_classes; i++)
      if (b.class_scores[i-1] > b.class_scores[pred-1]) pred = i;
  }
  ec.pred.multiclass = pred;

  for (uint32_t* i=b.labelset.begin; i!=b.labelset.end; ++i)
    b.class_scores[*i - 1] = 0.;

  //cerr << "truth=" << mc.label << " returning " << ec.pred.multiclass << endl << endl;

  /*
  if ((uint32_t)log(b.num_classes_popped) > (uint32_t)log(num_classes_popped_old))
    cerr << b.num_classes_popped << " classes popped, " << b.num_repeats << " repeats = " << (float)b.num_repeats/(float)b.num_classes_popped << "% repeats (of which " << (float)b.num_repeats_wrong/(float)b.num_classes_popped << "% is wrong), accounting for " << b.repeat_mass / b.class_mass << "% of the mass" << endl;
  */
  
  ec.l.multi = mc;
}

void predict(oas& b, base_learner& base, example& ec) {
  predict_ucs(b, base, ec, nullptr);
  //predict_noucs(b, base, ec);
}

void learn(oas& b, base_learner& base, example& ec)
{ //    verify_min_dfs(b, b.nodes[0]);
  MULTICLASS::label_t mc = ec.l.multi;
  v_array<weighted_node>* nodeset  = nullptr;
  if ((ec.l.multi.label != (uint32_t)-1) && (b.train_method != TRAIN_SINGLE_PATH)) {
    nodeset   = new v_array<weighted_node>();
    *nodeset  = v_init<weighted_node>();
  }
  
  predict_ucs(b,base,ec,nodeset);
  if(mc.label != (uint32_t)-1)  //if training the tree
  { uint32_t start_pred = ec.pred.multiclass;
    // train one-against-some
    if ((!b.predict_by_sum) && (b.MaxDequeues > 1)) {
      ec.l.simple = { 1.f, 1.f, 0.f };
      base.learn(ec, b.k + mc.label-1);
      ec.l.simple = { -1.f, 1.f, 0.f };
      for (uint32_t*lab=b.labelset.begin; lab!=b.labelset.end; ++lab)
        if (*lab != mc.label)
          base.learn(ec, b.k + *lab-1);
    }

    // train the tree
    if (b.train_method == TRAIN_SINGLE_PATH) {
      uint32_t class_index = 0;
      ec.l.simple = {FLT_MAX, 0.f, 0.f};
      uint32_t cn = 0;
      while(children(b, cn, class_index, mc.label, 1.0))
      { train_node(b, base, ec, cn, class_index, 1.0);
        cn = descend(b.nodes[cn], ec.pred.scalar);
      }

      b.nodes[cn].min_count += 1.0;
      update_min_count(b, cn);
    } else {
      for (weighted_node*wn = nodeset->begin; wn!=nodeset->end; ++wn) {
        uint32_t cn = wn->node;
        float    w  = wn->weight;
        uint32_t class_index = 0;
        ec.l.simple = {FLT_MAX, 0.f, 0.f};
        if (children(b, cn, class_index, mc.label, w))
          train_node(b, base, ec, cn, class_index, w);
        else {
          b.nodes[cn].min_count += w;
          update_min_count(b, cn);
        }
      }
    }
    
    ec.pred.multiclass = start_pred;
    ec.l.multi = mc;
  }

  if (nodeset != nullptr) {
    nodeset->delete_v();
    delete nodeset;
  }
}

void save_node_stats(oas& d)
{ FILE *fp;
  uint32_t i, j;
  double total;
  oas* b = &d;

  fp = fopen("atxm_debug.csv", "wt");

  for(i = 0; i < b->nodes.size(); i++)
  { fprintf(fp, "Node: %4d, Internal: %1d, Eh: %7.4f, n: %6f, \n", (int) i, (int) b->nodes[i].internal, b->nodes[i].Eh / b->nodes[i].n, b->nodes[i].n);

    fprintf(fp, "Label:, ");
    for(j = 0; j < b->nodes[i].preds.size(); j++)
    { fprintf(fp, "%6d,", (int) b->nodes[i].preds[j].label);
    }
    fprintf(fp, "\n");

    fprintf(fp, "Ehk:, ");
    for(j = 0; j < b->nodes[i].preds.size(); j++)
    { fprintf(fp, "%7.4f,", b->nodes[i].preds[j].Ehk / b->nodes[i].preds[j].nk);
    }
    fprintf(fp, "\n");

    total = 0.;

    fprintf(fp, "nk:, ");
    for(j = 0; j < b->nodes[i].preds.size(); j++)
    { fprintf(fp, "%6g,", b->nodes[i].preds[j].nk);
      total += b->nodes[i].preds[j].nk;
    }
    fprintf(fp, "\n");

    fprintf(fp, "max(lab:cnt:tot):, %3d,%6f,%7g,\n", (int) b->nodes[i].max_count_label, b->nodes[i].max_count, total);
    fprintf(fp, "left: %4d, right: %4d", (int) b->nodes[i].left, (int) b->nodes[i].right);
    fprintf(fp, "\n\n");
  }

  fclose(fp);
}

void finish(oas& b)
{ //save_node_stats(b);
  for (size_t i = 0; i < b.nodes.size(); i++)
    b.nodes[i].preds.delete_v();
  b.nodes.delete_v();
  free(b.class_scores);
  b.labelset.delete_v();
}

void save_load_tree(oas& b, io_buf& model_file, bool read, bool text)
{ if (model_file.files.size() > 0)
    { stringstream msg;
      msg << "k = " << b.k;
      bin_text_read_write_fixed(model_file,(char*)&b.k, sizeof(b.k), "", read, msg, text);

      msg << "num_classes = " << b.num_classes;
      bin_text_read_write_fixed(model_file,(char*)&b.num_classes, sizeof(b.num_classes), "", read, msg, text);

      msg << "nodes = " << b.nodes.size() << " ";
      uint32_t temp = (uint32_t)b.nodes.size();
      bin_text_read_write_fixed(model_file,(char*)&temp, sizeof(temp), "", read, msg, text);
    if (read)
      for (uint32_t j = 1; j < temp; j++)
        b.nodes.push_back(init_node());

    msg << "max predictors = " << b.max_predictors << " ";
    bin_text_read_write_fixed(model_file,(char*)&b.max_predictors, sizeof(b.max_predictors), "", read, msg, text);

    msg << "predictors_used = " << b.predictors_used << " ";
    bin_text_read_write_fixed(model_file,(char*)&b.predictors_used, sizeof(b.predictors_used), "", read, msg, text);

    msg << "swap_resist = " << b.swap_resist << "\n";
    bin_text_read_write_fixed(model_file,(char*)&b.swap_resist, sizeof(b.swap_resist), "", read, msg, text);

    for (size_t j = 0; j < b.nodes.size(); j++)
    { //Need to read or write nodes.
      node& n = b.nodes[j];

      msg << " parent = " << n.parent;
      bin_text_read_write_fixed(model_file,(char*)&n.parent, sizeof(n.parent), "", read, msg, text);

      uint32_t temp = (uint32_t)n.preds.size();

      msg << " preds = " << temp;
      bin_text_read_write_fixed(model_file,(char*)&temp, sizeof(temp), "", read, msg, text);
      if (read)
        for (uint32_t k = 0; k < temp; k++)
          n.preds.push_back(node_pred(1));

      msg << " min_count = " << n.min_count;
      bin_text_read_write_fixed(model_file,(char*)&n.min_count, sizeof(n.min_count), "", read, msg, text);

      msg << " internal = " << n.internal;
      bin_text_read_write_fixed(model_file,(char*)&n.internal, sizeof(n.internal), "", read, msg, text);

      if (n.internal)
        { msg << " base_predictor = " << n.base_predictor;
        bin_text_read_write_fixed(model_file,(char*)&n.base_predictor, sizeof(n.base_predictor), "", read, msg, text);

        msg << " left = " << n.left;
        bin_text_read_write_fixed(model_file,(char*)&n.left, sizeof(n.left), "", read, msg, text);

        msg << " right = " << n.right;
        bin_text_read_write_fixed(model_file,(char*)&n.right, sizeof(n.right), "", read, msg, text);

        msg << " norm_Eh = " << n.norm_Eh;
        bin_text_read_write_fixed(model_file,(char*)&n.norm_Eh, sizeof(n.norm_Eh), "", read, msg, text);

        msg << " Eh = " << n.Eh;
        bin_text_read_write_fixed(model_file,(char*)&n.Eh, sizeof(n.Eh), "", read, msg, text);

        msg << " n = "<< n.n << "\n";
        bin_text_read_write_fixed(model_file,(char*)&n.n, sizeof(n.n), "", read, msg, text);
      }
      else
        { msg << " max_count = " << n.max_count;
          bin_text_read_write_fixed(model_file,(char*)&n.max_count, sizeof(n.max_count), "", read, msg, text);
          msg << " max_count_label = "<< n.max_count_label <<"\n";
          bin_text_read_write_fixed(model_file,(char*)&n.max_count_label, sizeof(n.max_count_label), "", read, msg, text);
        }

      for (size_t k = 0; k < n.preds.size(); k++)
      { node_pred& p = n.preds[k];

        msg << "  Ehk = " << p.Ehk;
        bin_text_read_write_fixed(model_file,(char*)&p.Ehk, sizeof(p.Ehk), "", read, msg, text);

        msg << " norm_Ehk = " << p.norm_Ehk;
        bin_text_read_write_fixed(model_file,(char*)&p.norm_Ehk, sizeof(p.norm_Ehk), "", read, msg, text);

        msg << " nk = " << p.nk;
        bin_text_read_write_fixed(model_file,(char*)&p.nk, sizeof(p.nk), "", read, msg, text);

        msg << " label = " << p.label;
        bin_text_read_write_fixed(model_file,(char*)&p.label, sizeof(p.label), "", read, msg, text);

        msg << " label_count = "<< p.label_count << "\n";
        bin_text_read_write_fixed(model_file,(char*)&p.label_count, sizeof(p.label_count), "", read, msg, text);
      }
    }
  }
}

base_learner* oas_setup(vw& all)	//learner setup
{ if (missing_option<size_t, true>(all, "oas", "Use online tree for multiclass"))
    return nullptr;
  new_options(all, "Logarithmic Time Multiclass options")
      ("swap_resistance", po::value<uint32_t>(), "higher = more resistance to swap, default=4")
      ("tree_size", po::value<uint32_t>(), "change the size of the tree, default=K")
      ("evaluate_recall", "compute the recall of the tree, rather than the accuracy")
      ("max_mass", po::value<float>(), "stop popping once we've hit this amount of probability mass, default=0.999")
      ("max_dequeues", po::value<uint32_t>(), "stop popping once we've poped this many labels, default=tree_size")
      ("predict_by_sum", "predict using tree sum probability (rather than oas)")
      ("dequeue_leaves", "dequeue leaves rather than classes")
      ("train_single_path", "only train on the predicted path");
  add_options(all);

  po::variables_map& vm = all.vm;

  oas& data = calloc_or_throw<oas>();
  data.train_method = TRAIN_HIT_NODES;
  data.k = (uint32_t)vm["oas"].as<size_t>();
  data.num_classes = data.k;
  data.swap_resist = 4;
  data.log_MaxMassToConsider = log(0.999);

  if (vm.count("swap_resistance"))
    data.swap_resist = vm["swap_resistance"].as<uint32_t>();

  if (vm.count("train_single_path"))
    data.train_method = TRAIN_SINGLE_PATH;
  
  if (vm.count("tree_size")) {
    data.num_classes = data.k;
    data.k = vm["tree_size"].as<uint32_t>();
  }

  data.MaxDequeues = 2 * data.k;
  if (vm.count("max_mass"))
    data.log_MaxMassToConsider = log(vm["max_mass"].as<float>());
  if (vm.count("max_dequeues"))
    data.MaxDequeues = vm["max_dequeues"].as<uint32_t>();
  
  data.evaluate_recall = vm.count("evaluate_recall") > 0;
  data.predict_by_sum = vm.count("predict_by_sum") > 0;
  data.dequeue_leaves = vm.count("dequeue_leaves") > 0;

  // switch to logistic loss and link
  all.args.push_back("--loss_function"); all.args.push_back("logistic");
  all.args.push_back("--link");          all.args.push_back("logistic");

  data.num_repeats = 0; data.num_repeats_wrong = 0; data.num_classes_popped = 0;
  data.repeat_mass = 0.; data.class_mass = 0.;
  data.max_predictors = data.k - 1 + data.num_classes;
  init_tree(data);

  //data.Qptr = new Beam::beam<treenode>(MaxBeamSize, FLT_MAX, nullptr, false);
  new (&data.Q) pqueue<treenode>(2 * (uint32_t)log(data.num_classes));
  data.class_scores = calloc_or_throw<float>(data.num_classes);
  
  learner<oas>& l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.max_predictors);
  l.set_save_load(save_load_tree);
  l.set_finish(finish);

  return make_base(l);
}

}
