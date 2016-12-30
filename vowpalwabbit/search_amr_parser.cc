/*
  Copyright (c) by respective owners including Yahoo!, Microsoft, and
  individual contributors. All rights reserved.  Released under a BSD (revised)
  license as described in the file LICENSE.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "search_amr_parser.h"
#include "gd.h"
#include "cost_sensitive.h"
#include "label_dictionary.h"   // for add_example_namespaces_from_example
#include "vw.h"
#include "vw_exception.h"

#define val_namespace 100 // valency and distance feature space
#define offset_const 344429

using namespace std;

namespace AMRParserTask         {  Search::search_task task = { "amr_parser", run, initialize, finish, setup, nullptr};  }

struct gold_head_T
{ v_array<uint32_t> items;   // the is what we used to use
  v_array<uint32_t> count;   // count[i] == # of times i appears in items, for efficiency
  gold_head_T() { items = v_init<uint32_t>(); count = v_init<uint32_t>(); }
  void delete_v() { items.delete_v(); count.delete_v(); }
  void erase() { items.erase(); count.erase(); }
  inline uint32_t*& begin() { return items.begin(); }
  inline uint32_t*& end() { return items.end(); }
  inline bool contains(uint32_t x) { return (x < count.size()) && (count[x] > 0); }
  inline bool get_count(uint32_t x) { return (x < count.size()) ? count[x] : 0; }
  inline size_t size() { return items.size(); }
  inline uint32_t operator[](uint32_t i) { return items[i]; }
  void push_back(uint32_t x)
  { items.push_back(x);
    if (count.size() < x+1) {
      count.resize(x+4);
      count.end() = count.begin() + x + 1;
    }
    count[x] ++;
  }
};
  
std::ostream& operator<<(std::ostream& os, const gold_head_T& v) { return (os << v.items); }

struct task_data
{ example *ex;
  size_t amr_root_label;
  bool always_include_null_concept;
  bool use_gold_concepts;
  size_t max_hallucinate_in_a_row;
  bool disallow_swap;
  bool reverse_sentence;
  uint32_t amr_num_label;
  uint32_t amr_num_concept;
  v_array<uint32_t> valid_actions, action_loss, stack, temp, valid_action_temp, gold_concepts, concepts;
  v_array<v_array<uint32_t>> heads, gold_tags, tags;
  v_array<gold_head_T> gold_heads;
  v_array<action> gold_actions, gold_action_temp;
  v_array<pair<action, float>> gold_action_losses;
  v_array<v_array<pair<action,float>>> possible_concepts;
  v_array<uint32_t> children[6]; // [0]:num_left_arcs, [1]:num_right_arcs; [2]: leftmost_arc, [3]: second_leftmost_arc, [4]:rightmost_arc, [5]: second_rightmost_arc
  example * ec_buf[13];
  std::map<std::string, v_array<pair<action,float>>> word_to_concept;
  LabelDict::label_feature_map concept_to_features;
  v_array<v_array<uint32_t>> action_confusion_matrix;
};

namespace AMRParserTask
{
using namespace Search;

const action MAKE_CONCEPT         = 1;
const action REDUCE_RIGHT         = 2;
const action REDUCE_LEFT          = 3;
const action SWAP_REDUCE_RIGHT	  = 4;
const action SWAP_REDUCE_LEFT 	  = 5;
const action HALLUCINATE          = 6;
const action SHIFT 		  = 7;

const int NUM_ACTIONS = 7;
//const int NUM_LEARNERS = 7;
const int NULL_CONCEPT = 1;
const int NO_HEAD = 0; //this is not predicted and hence can be 0
const int NO_EDGE = 0; //this is not predicted and hence can be 0
const int ROOT = 0;
const int MAX_SENT_LEN = 300;


// TODO : Maybe replace by function from v_array.h ?
bool contains(v_array<uint32_t>& v, uint32_t x)
{
      if (v.empty())
           return false;
      if (find(v.begin(), v.end(), x) != v.end())
           return true;
      else
           return false;
}

uint32_t contains_idx(gold_head_T& v, uint32_t p,v_array<uint32_t>& exclude)
{ if (v.size() == 0)
    return v.size() + 10;
  for (size_t idx=0; idx<v.size(); idx++)
  { if (v[idx] == p)
    { if (!contains(exclude,idx))
        return idx;
      else
        cdbg << " Excluding " << idx << endl;
    }
  }
  return v.size()+10;
}

uint32_t get_count(v_array<uint32_t>& v, uint32_t x)
{ uint32_t ct = 0;
  for(size_t i=0; i<v.size(); i++)
    if(v[i] == x)
      ct += 1;
  return ct;
}

  inline bool contains_gh(gold_head_T& v, uint32_t x) { return v.contains(x); }
  inline uint32_t get_count_gh(gold_head_T& v, uint32_t x) { return v.get_count(x); }
  /*
    uint32_t c = v.get_count(x);
    uint32_t c2 = get_count(v.items, x);
    if (c != c2) cerr << '(' << c << ' ' << c2 << '|' << v.items << '/' << v.count << ')';
    return c2;
  }
  */
  //inline bool contains_gh(gold_head_T& v, uint32_t x) { return contains(v.items, x); }
  //inline uint32_t get_count_gh(gold_head_T& v, uint32_t x) { return get_count(v.items, x); }
  
  
// read a dictionary. keep AT MOST max_competitors concepts for any word. ALSO throw out anythign with count < min_count
size_t read_word_to_concept(string fname, std::map<std::string, v_array<pair<action,float>>>& dict, action& num_concept, size_t max_competitors=INT_MAX, float min_count=0.)
{ size_t max_confusion_set = 1;
  ifstream file(fname);
  assert(file.is_open());
  string str;
  string w;
  int f;
  float x;
  char c;
  while(getline(file, str))
  { istringstream ss(str);
    float total = 0.;
    ss >> w;
    v_array<pair<action,float>> me = v_init<pair<action,float>>();
    while (ss)
    { ss >> f;
      ss >> c;
      assert(c == ':');
      ss >> x;
      if (!ss) break;
      total += x;
      if (x >= min_count)
        me.push_back( make_pair(f,x) );
    }
    auto entry = dict.find(w);
    if (entry != dict.end())
    { //cerr << "warning: word '" << w << "' appears multiple times in dictionary " << fname << "; skipping later occurances" << endl;
      me.delete_v();
      continue;
    }
    if ((me.size() == 0) || (total <= 0.))
    { me.delete_v();
      continue;
    }
    // sort by default
    std::sort(me.begin(), me.end(),
              [](const pair<action,float> a, const pair<action,float> b) -> bool { return a.second > b.second; });
    if (me.size() > max_competitors)
      me.resize(max_competitors);
    for (pair<action,float>& p : me)
    { p.second /= total;
      num_concept = max(num_concept, p.first);
    }

    dict.insert( make_pair(w, me) );
    max_confusion_set = max(max_confusion_set, me.size());
  }
  return max_confusion_set;
}

void initialize(Search::search& sch, size_t& num_actions, po::variables_map& vm)
{ vw& all = sch.get_vw_pointer_unsafe();
  task_data *data = new task_data();
  data->action_loss.resize(NUM_ACTIONS+1);
  if (data->action_loss.size() == 0) data->action_loss.push_back(0);
  data->action_loss.end() = data->action_loss.begin() + NUM_ACTIONS + 1;
  data->ex = NULL;
  sch.set_task_data<task_data>(data);
  //sch.set_force_oracle(1);

  size_t max_competitors = INT_MAX;
  float  min_count = 0.;

  new_options(all, "AMR Parser Options")
      ("amr_root_label", po::value<size_t>(&(data->amr_root_label))->default_value(1), "Ensure that there is only one root in each sentence")
      ("amr_no_auto_null_concept", "by default all words can yield a null concept; turn on this flag to disable this")
      ("amr_num_label", po::value<uint32_t>(&(data->amr_num_label))->default_value(5), "Number of arc labels")
      ("amr_max_hallucinate", po::value<size_t>(&(data->max_hallucinate_in_a_row))->default_value(2), "Maximum number of hallucinations in a row to avoid infinite loops (def: 2)")
      ("amr_disallow_swap", "Turn off the swap actions")
      ("use_gold_concepts", po::value<bool>(&(data->use_gold_concepts))->default_value(false), "turn on this flag to use gold concepts at test time")
      ("amr_dictionary", po::value<string>(), "file to read word-to-concept dictionary from")
      ("amr_dictionary_max_competitors", po::value<size_t>(&max_competitors)->default_value(INT_MAX), "restrict concept sets to at most this many items (def: infinity)")
      ("amr_dictionary_min_count", po::value<float>(&min_count)->default_value(0.), "ignore concepts with count/value less than this number (def: 0)")
      ("amr_reverse_sentence", "parse the sentence backwards")
      ;
  add_options(all);

  check_option<size_t>(data->amr_root_label, all, vm, "amr_root_label", false, size_equal,
                       "warning: you specified a different value for --amr_root_label than the one loaded from regressor. proceeding with loaded value: ", "");
  check_option<uint32_t>(data->amr_num_label, all, vm, "amr_num_label", false, uint32_equal,
                         "warning: you specified a different value for --amr_num_label than the one loaded from regressor. proceeding with loaded value: ", "");

  if (vm.count("amr_dictionary") == 0)
    THROW("AMR parsing needs a word-to-concept dictionary; please specify --amr_dictionary");

  data->always_include_null_concept = vm.count("amr_no_auto_null_concept") == 0;
  data->ex = VW::alloc_examples(sizeof(polylabel), 1);
  data->ex->indices.push_back(val_namespace);
  for(size_t i=1; i<14; i++)
    data->ex->indices.push_back((unsigned char)i+'A');
  data->ex->indices.push_back(constant_namespace);
  data->disallow_swap = vm.count("amr_disallow_swap") > 0;

  size_t max_confusion_set = read_word_to_concept(vm["amr_dictionary"].as<string>(), data->word_to_concept, data->amr_num_concept, max_competitors, min_count);

  sch.set_num_learners({false, true, false, false, false, false, true});
  sch.ldf_alloc(max(MAX_SENT_LEN, max_confusion_set));

  data->possible_concepts = v_init<v_array<pair<action,float>>>();
  for (size_t i=0; i<MAX_SENT_LEN; i++)
    data->possible_concepts.push_back(v_init<pair<action,float>>());
  data->reverse_sentence = vm.count("amr_reverse_sentence") > 0;

  // features are:
  //    B - stack[-1]
  //    C - stack[-2]
  //    D - stack[-3]
  //    E - buffer[0]
  //    F - buffer[1]
  //    G - buffer[2]
  //    H - closest-on-left-child-of-stack[-1]
  //    I - 2nd-closest-on-left-child-of-stack[-1]
  //    J - closest-on-right-child-of-stack[-1]
  //    K - 2nd-closest-on-right-child-of-stack[-1]
  //    L - closest-on-left-child-of-buffer[0]
  //    M - 2nd-closest-on-left-child-of-buffer[1]
  //    N - nothing???
  //    d - valency features

  const char* pair[] = { "BE", "BC", "CE", "BD", "dB", "dC", "dD", "dE", "Bh", "dh" };
  vector<string> newpairs(pair, pair+10);
  
  //const char* pair[] = { "BC", "BE", "BB", "CC", "DD", "EE", "FF", "GG", "EF", "BH", "BJ", "EL", "dB", "dC", "dD", "dE", "dF", "dG", "dd",
  //                       "Bh", "Ch", "Eh", "Dh", "Fh", "Gh", "Hh", "Jh", "Lh", "dh" };
  //vector<string> newpairs(pair, pair+29);
  //const char* pair[] = {"Bh", "Ch", "Eh", "Dh", "Fh", "Gh", "Hh", "Jh", "Lh", "dh" };
  //vector<string> newpairs(pair, pair+10);
  all.pairs.swap(newpairs);
  //const char* triple[] = {"EFG", "BEF", "BCE", "BCD", "BEL", "ELM", "BHI", "BCC", "BEJ", "BEH", "BJK", "BEN"};
  //vector<string> newtriples(triple, triple+12);
  //all.triples.swap(newtriples);

  num_actions = max(data->amr_num_label, SHIFT);

  for (v_string& i : all.interactions)
    i.delete_v();
  all.interactions.erase();
  for (string& i : all.pairs)
    all.interactions.push_back(string2v_string(i));
  //for (string& i : all.triples)
  //all.interactions.push_back(string2v_string(i));
  sch.set_options(AUTO_CONDITION_FEATURES | NO_CACHING | IS_MIXED_LDF);

  sch.set_label_parser( COST_SENSITIVE::cs_label, [](polylabel&l) -> bool { return l.cs.costs.size() == 0; });
  //Action confusion matrix for debugging; row=gold_action col=predicted_action
  data->action_confusion_matrix.resize(NUM_ACTIONS);
  for (size_t i=0; i<NUM_ACTIONS; i++)
  { data->action_confusion_matrix[i].resize(NUM_ACTIONS);
    for (size_t j=0; j<NUM_ACTIONS; j++)
    {  data->action_confusion_matrix[i][j] = 0;
    }
  }
}

void finish(Search::search& sch)
{ task_data* data = sch.get_task_data<task_data>();
  if (sch.is_test() == true) //this is test time, so print action_confusion_matrix
  { cerr << "Action Confusion Matrix"<< endl;
    for (size_t i=0; i<NUM_ACTIONS; i++)
    { cerr << "action=" << i+1 << "  ";
      for (size_t j=0; j<NUM_ACTIONS; j++)
		cerr << data->action_confusion_matrix[i][j] << " ";
	  cerr << endl;
	}  
  }

  data->valid_actions.delete_v();
  data->valid_action_temp.delete_v();
  for (auto&x: data->gold_heads) x.delete_v();
  for (auto&x: data->gold_tags) x.delete_v();
  for (auto*x = data->heads.begin(); x != data->heads.end_array; ++x) x->delete_v();
  for (auto*x = data->tags.begin(); x != data->tags.end_array; ++x) x->delete_v();
  for (auto&x : data->possible_concepts) x.delete_v();
  data->possible_concepts.delete_v();
  data->gold_heads.delete_v();
  data->gold_tags.delete_v();
  data->gold_concepts.delete_v();
  data->stack.delete_v();
  data->heads.delete_v();
  data->tags.delete_v();
  data->concepts.delete_v();
  data->temp.delete_v();
  data->action_loss.delete_v();
  data->gold_actions.delete_v();
  data->gold_action_losses.delete_v();
  data->gold_action_temp.delete_v();
  VW::dealloc_example(COST_SENSITIVE::cs_label.delete_label, *data->ex);
  free(data->ex);
  LabelDict::free_label_features(data->concept_to_features);
  data->concept_to_features.delete_v();
  for (size_t i=0; i<6; i++) data->children[i].delete_v();
  delete data;
}

void inline add_feature(example& ex, uint64_t idx, unsigned char ns, uint64_t mask, uint64_t multiplier, bool audit=false)
{ ex.feature_space[(int)ns].push_back(1.0f, (idx * multiplier) & mask);
}

void add_all_features(example& ex, example& src, unsigned char tgt_ns, uint64_t mask, uint64_t multiplier, uint64_t offset, bool audit=false)
{ features& tgt_fs = ex.feature_space[tgt_ns];
  for (namespace_index ns : src.indices)
    if(ns != constant_namespace) // ignore constant_namespace
        for (feature_index i : src.feature_space[ns].indicies)
            tgt_fs.push_back(1.0f, ((i / multiplier + offset) * multiplier) & mask );
}

void inline reset_ex(example *ex)
{ ex->num_features = 0;
  ex->total_sum_feat_sq = 0;
  for (features& fs : *ex)
    fs.erase();
}

// arc-hybrid System.
size_t transition_hybrid(Search::search& sch, uint64_t a_id, uint32_t idx, uint32_t t_id)
{ task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &stack=data->stack, &concepts=data->concepts;
  v_array<v_array<uint32_t>> &heads=data->heads, &tags = data->tags;
  v_array<uint32_t> *children = data->children;
  if (a_id == MAKE_CONCEPT)
  { concepts[idx] = t_id;
    if (t_id == NULL_CONCEPT)
    {  heads[idx].push_back(NO_HEAD);
       tags[idx].push_back(NO_EDGE);
       return idx+1;
    }
    else
    {   return idx;
    }
  }
  else if (a_id == SHIFT)
  { stack.push_back(idx);
    return idx+1;
  }
  else if (a_id == REDUCE_RIGHT)
  { uint32_t last   = stack.last();
    uint32_t   hd     = stack[ stack.size() - 2 ];
    heads[last].push_back(hd);
    children[5][hd] = children[4][hd];
    children[4][hd] = last;
    children[1][hd] ++;
    tags[last].push_back(t_id);

    assert(! stack.empty());
    stack.pop();
    return idx;
  }
  else if (a_id == REDUCE_LEFT)
  { size_t last    = stack.last();
    heads[last].push_back(idx);
    children[3][idx] = children[2][idx];
    children[2][idx] = last;
    children[0][idx] ++;
    tags[last].push_back(t_id);

    assert(! stack.empty());
    stack.pop();
    return idx;
  }
  else if (a_id == SWAP_REDUCE_RIGHT)
  {
    size_t last = stack.last();
    stack.pop();
    size_t second_last = stack.last();
    stack.pop();
    size_t third_last = stack.last();
    stack.pop();
    heads[last].push_back(third_last);
    children[5][third_last] = children[4][third_last];
    children[4][third_last] = last;
    children[1][third_last] ++;
    tags[last].push_back(t_id);

    stack.push_back(third_last);
    stack.push_back(second_last);
    return idx;
  }
  else if (a_id == SWAP_REDUCE_LEFT)
  {
    size_t last = stack.last();
    stack.pop();
    size_t second_last = stack.last();
    stack.pop();
    heads[second_last].push_back(idx);
    children[3][idx] = children[2][idx];
    children[2][idx] = second_last;
    children[0][idx] ++;
    tags[second_last].push_back(t_id);

    stack.push_back(last);
    return idx;
   }
   else if (a_id == HALLUCINATE)
   {
     stack.push_back(t_id);
     return idx;
   }
  assert(false);
  THROW("transition_hybrid failed on a_id=" << a_id);
}

#define EC(i) ec[data->reverse_sentence ? n-(i)-1 : i]
  
void extract_features(Search::search& sch, uint32_t idx,  vector<example*> &ec)
{ vw& all = sch.get_vw_pointer_unsafe();
  task_data *data = sch.get_task_data<task_data>();
  reset_ex(data->ex);
  uint64_t mask = sch.get_mask();
  uint64_t multiplier = all.wpp << all.weights.stride_shift();
  v_array<uint32_t> &stack = data->stack, *children = data->children, &temp=data->temp;
  //v_array<v_array<uint32_t>> &tags = data->tags;
  example **ec_buf = data->ec_buf;
  example &ex = *(data->ex);

  size_t n = ec.size();
  bool empty = stack.empty();
  size_t last = empty ? 0 : stack.last();

  for(size_t i=0; i<13; i++)
    ec_buf[i] = nullptr;

  // feature based on the top three examples in stack ec_buf[0]: s1, ec_buf[1]: s2, ec_buf[2]: s3
  for(size_t i=0; i<3; i++)
    ec_buf[i] = (stack.size()>i && *(stack.end()-(i+1))!=0) ? EC(*(stack.end()-(i+1))-1) : 0;

  // features based on examples in string buffer ec_buf[3]: b1, ec_buf[4]: b2, ec_buf[5]: b3
  for(size_t i=3; i<6; i++)
  { ec_buf[i] = (idx+(i-3)-1 < n) ? EC(idx+i-3-1) : 0;
  }

  // features based on the leftmost and the rightmost children of the top element stack ec_buf[6]: sl1, ec_buf[7]: sl2, ec_buf[8]: sr1, ec_buf[9]: sr2;
  for(size_t i=6; i<10; i++)
    if (!empty && last != 0&& children[i-4][last]!=0)
      ec_buf[i] = EC(children[i-4][last]-1);

  // features based on leftmost children of the top element in bufer ec_buf[10]: bl1, ec_buf[11]: bl2
  for(size_t i=10; i<12; i++)
    ec_buf[i] = (idx <=n && children[i-8][idx]!=0) ? EC(children[i-8][idx]-1) : 0;
  ec_buf[12] = (stack.size()>1 && *(stack.end()-2)!=0 && children[2][*(stack.end()-2)]!=0) ? EC(children[2][*(stack.end()-2)]-1) : 0;

  // unigram features
  for(size_t i=0; i<13; i++)
  { uint64_t additional_offset = (uint64_t)(i*offset_const);
    if (!ec_buf[i])
      add_feature(ex, (uint64_t) 438129041 + additional_offset, (unsigned char)((i+1)+'A'), mask, multiplier);
    else
      add_all_features(ex, *ec_buf[i], 'A'+(unsigned char)(i+1), mask, multiplier, additional_offset, false);
  }

  // Other features
  temp.resize(10);
  temp[0] = empty ? 0: (idx >n? 1: 2+min(5, idx - (uint64_t)last));
  temp[1] = empty? 1: 1+min(5, children[0][last]);
  temp[2] = empty? 1: 1+min(5, children[1][last]);
  temp[3] = idx>n? 1: 1+min(5 , children[0][idx]);
  //for(size_t i=4; i<8; i++)
  //  temp[i] = (!empty && children[i-2][last]!=0)?tags[children[i-2][last]]:15;
  //for(size_t i=8; i<10; i++)
  //  temp[i] = (idx <=n && children[i-6][idx]!=0)? tags[children[i-6][idx]] : 15;

  uint64_t additional_offset = val_namespace*offset_const;
  for(size_t j=0; j< 10; j++)
  {
    additional_offset += j* 1023;
    add_feature(ex, temp[j]+ additional_offset , val_namespace, mask, multiplier);
  }
  size_t count=0;
  for (features fs : *data->ex)
    { fs.sum_feat_sq = (float) fs.size();
      count+= fs.size();
    }

  size_t new_count;
  float new_weight;
  INTERACTIONS::eval_count_of_generated_ft(all, *data->ex, new_count, new_weight);

  data->ex->num_features = count + new_count;
  data->ex->total_sum_feat_sq = (float) count + new_weight;
}

void get_valid_actions(v_array<uint32_t> & valid_action, uint64_t idx, uint64_t n, uint64_t stack_depth, uint64_t state, v_array<uint32_t> &concepts, bool hallucinate_okay, v_array<v_array<uint32_t>> &tags, bool disallow_swap)
{ valid_action.erase();
  if(idx<=n && concepts[idx] == 0)
    valid_action.push_back( MAKE_CONCEPT );
  if(stack_depth >=2)
    valid_action.push_back( REDUCE_RIGHT );
  if(stack_depth >=1 && state!=0 && idx<=n && concepts[idx] != 0)
    valid_action.push_back( REDUCE_LEFT );
  if (! disallow_swap)
  { if(stack_depth >=3)
      valid_action.push_back( SWAP_REDUCE_RIGHT );
    if(stack_depth >=2 && state!=0 && idx<=n && concepts[idx] != 0)
      valid_action.push_back( SWAP_REDUCE_LEFT);
  }
  if(stack_depth >=1 && state!=0 && hallucinate_okay)
    for (uint32_t i=1; i<idx; i++)
      if (tags[i].size() > 0)
      { valid_action.push_back( HALLUCINATE);
        break;
      }
  if(idx<=n && concepts[idx] != 0)
    valid_action.push_back( SHIFT );
}

bool is_valid(uint64_t action, v_array<uint32_t> valid_actions)
{ for(size_t i=0; i< valid_actions.size(); i++)
    if(valid_actions[i] == action)
      return true;
  return false;
}

void get_gold_actions(Search::search &sch, uint32_t idx, uint64_t n, v_array<action>& gold_actions, bool& hallucinate_okay)
{ task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &action_loss = data->action_loss, &stack = data->stack, &valid_actions=data->valid_actions;
  v_array<v_array<uint32_t>> &heads=data->heads;
  v_array<gold_head_T>& gold_heads = data->gold_heads;
  size_t size = stack.size();
  size_t last = (size==0) ? 0 : stack.last();

  if (is_valid(REDUCE_LEFT,valid_actions) && contains_gh(gold_heads[last], idx))
  { gold_actions.push_back(REDUCE_LEFT);
    cdbg << "RET RL" << endl;
    return;
  }

  if (is_valid(SHIFT,valid_actions) &&( stack.empty() || contains_gh(gold_heads[idx], last))) // becoz if we take any ohter action, we will lose this edge
  { gold_actions.push_back(SHIFT);
    cdbg << "RET SH" << endl;
    return;
  }

  if (is_valid(HALLUCINATE,valid_actions))
  { for (uint32_t i=1; i<idx; i++)
    { if ((get_count_gh(gold_heads[i], last) > get_count(heads[i], last)) && !(v_array_contains(stack, i)))
      { gold_actions.push_back(HALLUCINATE);
        hallucinate_okay = true;
        cdbg << "RET HALLUCINATE" << endl;
        return;
      }
    }
  }

  for(size_t i = 1; i<=NUM_ACTIONS ; i++)
  {  action_loss[i] = (is_valid(i,valid_actions))?0:100;
     //cdbg << "action_loss " << action_loss[i] << endl;
  }
  action_loss[HALLUCINATE] = 100; // We don't want to hallucinate any other time

  //Losses for SHIFT action

  if (size >= 3)
  { for(size_t i = 0; i<size-2; i++)
      if(idx <=n && (contains_gh(gold_heads[stack[i]], idx) || contains_gh(gold_heads[idx], stack[i])))
        action_loss[SHIFT] += 1; // Edges to and from idx to anything in stack before size-2 is lost
  }
  if(size>0 && contains_gh(gold_heads[last], idx))
    action_loss[SHIFT] += 1; //we can't do REDUCE_LEFT anymore
  if(size>=2 && contains_gh(gold_heads[stack[size-2]], idx))
    action_loss[SHIFT] += 1; //we can't do SWAP_REDUCE_LEFT anymore

  //Losses for REDUCE_LEFT action

  if(size>0 && !contains_gh(gold_heads[last], idx)) //If no such edge exists then add loss
    action_loss[REDUCE_LEFT] +=1;

  if(size>0)
  { //Edge to and from last to anything in buffer is lost
    for(size_t i = idx+1; i<=n; i++)
      if(contains_gh(gold_heads[i], last) || (contains_gh(gold_heads[last], i)))
        action_loss[REDUCE_LEFT] +=1;
    if(idx <=n && contains_gh(gold_heads[idx], last))
      action_loss[REDUCE_LEFT] +=1;
  }
  if(size>=2 && contains_gh(gold_heads[last], stack[size-2])) //i.e. we can't do REDUCE_RIGHT anymore
    action_loss[REDUCE_LEFT] += 1;
  if(size>=3 && contains_gh(gold_heads[last], stack[size-3])) //i.e. we can't do SWAP_REDUCE_RIGHT anymore
    action_loss[REDUCE_LEFT] += 1;

  //Losses for REDUCE_RIGHT action

  if(size>1 && !contains_gh(gold_heads[last], stack[size-2])) //If no such edge exists then add loss
    action_loss[REDUCE_RIGHT] +=1;

  if(size>0)
  { //Edge to and from last to anything in buffer is lost
    for(size_t i = idx; i<=n; i++)
      if(contains_gh(gold_heads[i], last) || (contains_gh(gold_heads[last], i)))
        action_loss[REDUCE_RIGHT] +=1;
  }

  if(size>=2 && contains_gh(gold_heads[last], idx)) //i.e. we can't do REDUCE_LEFT anymore
    action_loss[REDUCE_RIGHT] += 1;
  if(size>=3 && contains_gh(gold_heads[last], stack[size-3])) //i.e. we can't do SWAP_REDUCE_RIGHT anymore
    action_loss[REDUCE_RIGHT] += 1;


  //Losses for SWAP_REDUCE_LEFT action

  if(size>=2 && !contains_gh(gold_heads[stack[size-2]], idx)) //If no such edge exists then add loss
   action_loss[SWAP_REDUCE_LEFT] +=1;

  if(size>=2)
  { //Edge to and from second_last to anything in buffer is lost
    size_t second_last = stack[size-2];
    for(size_t i = idx; i<=n; i++)
      if(contains_gh(gold_heads[i], second_last) || (contains_gh(gold_heads[second_last], i)))
        action_loss[SWAP_REDUCE_LEFT] +=1;
  }

  if(size>=2 && contains_gh(gold_heads[last], stack[size-2])) //i.e. we can't do REDUCE_RIGHT anymore
    action_loss[SWAP_REDUCE_LEFT] += 1;
  
  /*
  if(size >= 2)
  { if(size>=3 && contains_gh(gold_heads[stack[size-2]], stack[size-3]))
      action_loss[SWAP_REDUCE_LEFT] +=1; //we can't do REDUCE_RIGHT when second_last comes on top of stack anymore
    if(size>=4 && contains_gh(gold_heads[stack[size-2]], stack[size-4]))
      action_loss[SWAP_REDUCE_LEFT] +=1; //we can't do SWAP_REDUCE_RIGHT when second_last comes on top of stack anymore

  }
  */

  //Losses for SWAP_REDUCE_RIGHT action

  if(size>=3 && !contains_gh(gold_heads[last], stack[size-3])) //If no such edge exists then add loss
    action_loss[SWAP_REDUCE_RIGHT] +=1;

  if(size>0)
  { //Edge to and from last to anything in buffer is lost
    for(size_t i = idx; i<=n; i++)
      if(contains_gh(gold_heads[i], last) || (contains_gh(gold_heads[last], i)))
        action_loss[SWAP_REDUCE_RIGHT] +=1;
  }

  if(size>=2 && contains_gh(gold_heads[last], stack[size-2])) //i.e. we can't do REDUCE_RIGHT anymore
    action_loss[SWAP_REDUCE_RIGHT] += 1;


  // return the best actions
  size_t best_action = 1;
  for(size_t i=1; i<NUM_ACTIONS; i++) //Exclude SHIFT for now
    if(action_loss[i] < action_loss[best_action])
    { best_action= i;
      gold_actions.erase();
      gold_actions.push_back((uint32_t)i);
    }
    else if (action_loss[i] == action_loss[best_action])
    { gold_actions.push_back(i);
    }
 
  /*
  if (action_loss[MAKE_CONCEPT] == action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back(MAKE_CONCEPT);
  }
  else if (action_loss[SWAP_REDUCE_LEFT] == action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back(SWAP_REDUCE_LEFT);
  }
  else if (action_loss[SWAP_REDUCE_RIGHT] == action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back(SWAP_REDUCE_RIGHT);
  }
  else if (action_loss[REDUCE_LEFT] == action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back(REDUCE_LEFT);
  }
  else if (action_loss[REDUCE_RIGHT] == action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back(REDUCE_RIGHT);
  }
  */ 
  // return SHIFT only if no other action is better than SHIFT
  if (action_loss[SHIFT] < action_loss[best_action])
  { gold_actions.erase();
    gold_actions.push_back((uint32_t)SHIFT);
  }
  //if (sch.at_learning_point())
  //  cerr << endl << "AL = " << action_loss;
}

void get_word_possible_concepts(task_data& data, v_array<pair<action,float>>& possible_concepts, v_array<char>& word)
{ string s = string(word.begin(), word.size());
  possible_concepts.erase();
  auto entry = data.word_to_concept.find(s);
  if (entry == data.word_to_concept.end())
  { //cerr << "warning: word '" << s << "' not found in word-to-concept dictionary" << endl;
    possible_concepts.push_back(make_pair(NULL_CONCEPT,1.0));
    return;
  }
  bool got_null = false;
  for (pair<action,float>& af : entry->second)
  { possible_concepts.push_back(af);
    if (af.first == NULL_CONCEPT) got_null = true;
  }
  if (data.always_include_null_concept && !got_null)
    possible_concepts.push_back(make_pair(NULL_CONCEPT,1e-6));
}

template<class T, T(*init)()>
void reverse_sentence(v_array<uint32_t>&concepts, v_array<v_array<uint32_t>>& tags, v_array<T>&heads, uint32_t n)
{ for (size_t _i=0; _i<n/2; _i++)
  { size_t i = _i+1;
    size_t j = n-_i;
    uint32_t tmp_c = concepts[i]; concepts[i] = concepts[j]; concepts[j] = tmp_c;
    v_array<uint32_t> tmp_t = tags[i]; tags[i] = tags[j]; tags[j] = tmp_t;
    T tmp_h = heads[i]; heads[i] = heads[j]; heads[j] = tmp_h;
  }
  T tmp = init();
  for (size_t i=1; i<=n; i++)
  { for (uint32_t j : heads[i])
    { uint32_t k = (j == 0) ? 0 : n-j+1;
      tmp.push_back(k);
    }
    heads[i].erase();
    for (uint32_t j : tmp)
      heads[i].push_back(j);
    tmp.erase();
  }
  tmp.delete_v();
}

gold_head_T init_gold_head_T() { return gold_head_T(); }
  
void setup(Search::search& sch, vector<example*>& ec)
{ task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &gold_concepts=data->gold_concepts, &concepts=data->concepts;
  v_array<v_array<uint32_t>> &heads=data->heads, &gold_tags=data->gold_tags, &tags=data->tags;
  v_array<gold_head_T>& gold_heads = data->gold_heads;
  size_t n = ec.size();
  if (n >= MAX_SENT_LEN)
    THROW("sentence too long, length=" << n << ", but MAX_SENT_LEN=" << MAX_SENT_LEN);
  v_array<uint32_t> empty_array = v_init<uint32_t>();
  gold_head_T empty_gold_head;

  for (size_t i=0; i<n; i++)
    get_word_possible_concepts(*data,
                               data->possible_concepts[i],
                               EC(i)->tag);

  for (auto*x = data->heads.begin(); x != data->heads.end_array; ++x) x->delete_v();
  for (auto*x = data->tags.begin(); x != data->tags.end_array; ++x) x->delete_v();
  heads.resize(n+1);
  tags.resize(n+1);

  heads[0] = v_init<uint32_t>();
  tags[0] = v_init<uint32_t>();

  concepts.resize(n+1);

  for (auto&x: gold_heads) x.delete_v();
  gold_heads.erase();
  gold_heads.push_back(empty_gold_head);
  gold_heads[0].push_back(0);

  for (auto&x: gold_tags) x.delete_v();
  gold_tags.erase();
  gold_tags.push_back(empty_array);
  gold_tags[0].push_back(0);

  gold_concepts.erase();
  gold_concepts.push_back(0);

  for (size_t i=0; i<n; i++)
  { v_array<COST_SENSITIVE::wclass>& costs = ec[i]->l.cs.costs;
    gold_head_T head;
    v_array<uint32_t> tag = v_init<uint32_t>();
    size_t h, t, concept;
    h = (costs.size() == 0) ? 0 : costs[0].class_index;
    t = (costs.size() <= 1) ? (uint64_t)data->amr_root_label : costs[1].class_index;
    concept  = (costs.size() <= 2) ? 0 : costs[2].class_index;
    if (t > data->amr_num_label)
      THROW("invalid label " << t << " which is > num_label=" << data->amr_num_label);
    if (concept > data->amr_num_concept)
      THROW("invalid concept " << concept << " which is > num_concept=" << data->amr_num_concept);
    head.push_back(h);
    tag.push_back(t);
    for (size_t j=3; j<costs.size(); j+=2)
    { h = costs[j].class_index;
      t = costs[j+1].class_index;
      if (t > data->amr_num_label)
        THROW("invalid label " << t << " which is > num actions=" << data->amr_num_label);
      head.push_back(h);
      tag.push_back(t);
    }
    cdbg << "head " << head[0] << endl;
    cdbg << "tag " << tag[0] << endl;
    gold_heads.push_back(head);
    gold_tags.push_back(tag);
    gold_concepts.push_back(concept);
    heads[i+1] = v_init<uint32_t>();
    //heads[i+1].push_back(0);
    tags[i+1] = v_init<uint32_t>();
    //tags[i+1].push_back(-1);
    concepts[i+1] = 0;

  }
  for(size_t i=0; i<6; i++)
    data->children[i].resize(n+(size_t)1);
  if (data->reverse_sentence)
    reverse_sentence<gold_head_T, init_gold_head_T>(gold_concepts, gold_tags, gold_heads, n);
}

float smatch_loss(Search::search& sch, uint64_t n)
{
  task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &gold_concepts=data->gold_concepts, &concepts=data->concepts;
  v_array<v_array<uint32_t>> &heads=data->heads, &gold_tags=data->gold_tags, &tags = data->tags;
  v_array<gold_head_T>& gold_heads = data->gold_heads;

  //if (data->reverse_sentence)
  //  reverse_sentence<v_array<uint32_t>, v_init<uint32_t>>(concepts, tags, heads, n);
  
  float correct = 0,pred = 0,gold = 0;
  // First, we calculate score of concepts - we don't care about root
  for (size_t i=1;i<=n;i++) {

    if (gold_concepts[i] != NULL_CONCEPT)
        gold += 1;
    if (concepts[i] != NULL_CONCEPT)
        pred += 1;
    if (gold_concepts [i] != NULL_CONCEPT && concepts[i] != NULL_CONCEPT && gold_concepts[i] == concepts[i])
        correct += 1;
  }

  /*
  heads.end() = heads.begin() + n+1;
  concepts.end() = concepts.begin() + n+1;
  tags.end() = tags.begin() + n+1;

  cerr << "gold_heads = " << gold_heads << endl;
  cerr << "     heads = " <<      heads << endl;
  cerr << "gold_tags = " << gold_tags << endl;
  cerr << "     tags = " <<      tags << endl;
  cerr << "gold_concepts = " << gold_concepts << endl;
  cerr << "     concepts = " <<      concepts << endl;
  */
  
  cdbg << "Gold " << gold << endl;
  cdbg << "Pred  " << pred << endl;
  cdbg << "Correct " << correct << endl;
  cdbg << "Conceptend" << endl;

  // Then relations
  // For each
  v_array<uint32_t> exclude = v_init<uint32_t>();
  for (size_t i =1;i<=n;i++)
  {
    cdbg << "Gold heads size " << gold_heads[i].size() << endl;
    cdbg << "Pred heads size " << heads[i].size() << endl;
    for (size_t j =0;j<gold_heads[i].size();j++)
      gold += 1;
    cdbg << "Printing heads " << i << endl;
    for (size_t j =0;j<heads[i].size();j++)
    {
      pred += 1;
      uint32_t p = contains_idx(gold_heads[i],heads[i][j],exclude);
      cdbg << "The function " << p << endl;
      if (p != gold_heads[i].size()+ 10) {

        cdbg << "Pred tag " << tags[i][j] << endl;
        cdbg << "Gold tag " << gold_tags[i][p] << endl;
        if (tags[i][j] == gold_tags[i][p])
        {
          correct += 1;
          exclude.push_back(p);
        }

      }
    }
    exclude.erase();
  }
  cdbg << "Gold " << gold << endl;
  cdbg << "Pred  " << pred << endl;
  cdbg << "Correct " << correct << endl;
  cdbg << "NewEnd" << endl;

  //return max(0., gold - correct + pred - correct);
  
  float p = (pred > 0) ? correct/pred : 0;
  float r = (gold > 0) ? correct/gold : 0;
  float smatch = (p+r > 0) ? 1 -(2*p*r/(p+r)): 1;

  return smatch;
}

void inline add_feature_val(example& ex, uint64_t idx, float val, unsigned char ns, uint64_t mask, uint64_t multiplier, bool audit=false)
{ ex.feature_space[(int)ns].push_back(val, (idx * multiplier) & mask);
  ex.num_features += 1;
  ex.total_sum_feat_sq += val*val;
}
  
void run(Search::search& sch, vector<example*>& ec)
{ task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &stack=data->stack, &valid_actions=data->valid_actions, &gold_concepts=data->gold_concepts, &concepts=data->concepts;
  v_array<v_array<uint32_t>> &heads=data->heads, &gold_tags=data->gold_tags, &tags=data->tags;
  v_array<gold_head_T>& gold_heads = data->gold_heads;
  v_array<action> &gold_actions = data->gold_actions;
  v_array<v_array<pair<action,float>>>& possible_concepts = data->possible_concepts;
  v_array<uint32_t> gold_ids = v_init<uint32_t>();
  
  LabelDict::label_feature_map& concept_to_features = data->concept_to_features;
  LabelDict::free_label_features(data->concept_to_features);
  concept_to_features.clear(); // erase current set of concepts
  uint64_t n = (uint64_t) ec.size();

  if (data->reverse_sentence)
    for (size_t i=0; i<n/2; i++)
    { example* tmp = ec[i];
      ec[i] = ec[n-i-1];
      ec[n-i-1] = tmp;
    }
  
  if (n >= MAX_SENT_LEN)
    THROW("sentence too long, length=" << n << ", but MAX_SENT_LEN=" << MAX_SENT_LEN);
  //cdbg << "n " << n << endl;
  stack.erase();
  for (size_t i=0; i<=n; i++)
  { concepts[i] = 0;
    heads[i].delete_v();
    tags[i].delete_v();
    cdbg << "gold_tags[i] " << gold_tags[i][0] << endl;
  }
  //cdbg << "stack_size" << stack.size() << endl;
  stack.push_back(ROOT); //if ROOT is pushed into stack then what prevents an AMR from having two roots?
  for(size_t i=0; i<6; i++)
    for(size_t j=0; j<n+1; j++)
      data->children[i][j] = 0;

  vw& all = sch.get_vw_pointer_unsafe();
  uint64_t mask = sch.get_mask();  
  uint64_t multiplier = all.wpp << all.weights.stride_shift();

  int count=1;
  size_t idx = 1;
  Search::predictor P(sch, (ptag) 0);

  v_array<action> valid_tags = v_init<action>();
  for (action a=1; a<data->amr_num_label; a++)
    valid_tags.push_back(a);

  size_t num_hallucinate_in_a_row = 0;
  
  //cdbg << "stack_size" << stack.size() << endl;
  while(stack.size()>1 || idx <= n)
  { bool extracted_features = false;
    if(sch.predictNeedsExample())
    { extract_features(sch, idx, ec);
      extracted_features = true;
    }
    bool hallucinate_okay = num_hallucinate_in_a_row < data->max_hallucinate_in_a_row;
    get_valid_actions(valid_actions, idx, n, (uint64_t) stack.size(), stack.empty() ? 0 : stack.last(), concepts, hallucinate_okay, tags, data->disallow_swap);

    cdbg << "VALID ACTIONS " << valid_actions << endl;
    cdbg << "idx " << idx << endl;
    cdbg << "stack_size " << stack.size() << endl;
    size_t a_id = 0, t_id = 0;
    bool need_to_hallucinate = false;
    gold_actions.erase();
    //if (sch.predictNeedsReference())
    get_gold_actions(sch, idx, n, gold_actions, need_to_hallucinate);
    if ((!sch.is_test()) && need_to_hallucinate && (!hallucinate_okay) && (num_hallucinate_in_a_row < 10)) // strict upper bound of 10
      valid_actions.push_back(HALLUCINATE);

    a_id= P.set_tag((ptag) count)
           .set_input(*(data->ex))
           .set_oracle(gold_actions)
           .set_allowed(valid_actions)
           .set_condition_range(count-1, sch.get_history_length(), 'p')
           .set_learner_id(0)
           .predict();
    
    if (sch.is_test() == true) //this is test time, so update action_confusion_matrix
    { //if (data->use_gold_concepts && contains(gold_actions, MAKE_CONCEPT))
      //  a_id = MAKE_CONCEPT;
     
      //cerr << "GOLD ACTIONS " << gold_actions << endl;
      if (contains(gold_actions, a_id))
        data->action_confusion_matrix[a_id-1][a_id-1] += 1;
	  else
      { for (size_t i=0; i<gold_actions.size(); i++)  
	      data->action_confusion_matrix[gold_actions[i]-1][a_id-1] += 1;
      }
    } 

    if (a_id == HALLUCINATE) num_hallucinate_in_a_row ++;
    else num_hallucinate_in_a_row = 0;
    cdbg << "PREDICTED ACTION " << a_id << ", hallucinate_okay=" << hallucinate_okay << ", need_to_hallucinate=" << need_to_hallucinate << ", num_hallucinate_in_a_row=" << num_hallucinate_in_a_row << endl;
    count++;
    if ((!extracted_features) && sch.predictNeedsExample())
      extract_features(sch, idx, ec);

    if (a_id == MAKE_CONCEPT)
    { // for LDF
      P.erase_oracles();
      //cerr << "possible_concepts[" << idx-1 << "] = " << possible_concepts[idx-1] << endl;
      for (size_t i = 0; i < possible_concepts[idx-1].size(); i++)
      { assert(i < MAX_SENT_LEN);
        uint32_t concept = possible_concepts[idx-1][i].first;
        float    concept_weight = possible_concepts[idx-1][i].second;
        example* ldf_ec = sch.ldf_example(i);
        if (sch.predictNeedsExample())
        { VW::clear_example_data(*ldf_ec);
          VW::copy_example_data(false, ldf_ec, data->ex);
          VW::offset_example_indices(ldf_ec, sch.get_stride_shift(), 28904713, 4832917 * (uint64_t)concept);
          char ns = ' ';
          ldf_ec->indices.push_back(ns);
          add_feature_val(*ldf_ec, 481048517, 1., ns, mask, multiplier);
          add_feature_val(*ldf_ec, 348190573, -log(concept_weight), ns, mask, multiplier);
          if (i == 0)
            add_feature_val(*ldf_ec, 4738051797, 1., ns, mask, multiplier);
        }
        sch.ldf_set_label(i, concept);
        if (concept == gold_concepts[idx])
          P.set_oracle(i);
      }
      t_id = P.set_tag((ptag)count)
          .set_input(sch.ldf_example(), possible_concepts[idx-1].size())
          .set_condition_range(count-1, sch.get_history_length(), 'p')
          .set_learner_id(a_id)
          .predict();
      cdbg << "make_concept id=" << t_id;
      t_id = sch.ldf_get_label(t_id);
      cdbg << ", label=" << t_id << endl;

	  //if (data->use_gold_concepts && sch.is_test())
      //  t_id = gold_concepts[idx];
      /* original
      uint32_t gold_concept = gold_concepts[idx];
      t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_concept)
              .set_max_allowed(data->amr_num_concept)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      cdbg << "t_id " << t_id << endl;
      */
      // for later hallucinations, mark a memory of the features at this concept
          //LabelDict::set_label_features(concept_to_features, idx, *(data->ex));  // TODO: is idx the right place? -- Yes
    }
    else if (a_id == REDUCE_LEFT)
    { uint32_t gold_label = 0;
      uint32_t last = stack.last();
      for (size_t j=0; j< gold_heads[last].size(); j++)
        if (gold_heads[last][j] == idx)
          gold_label = gold_tags[last][j];
      if(gold_label == 0)
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .erase_oracles()
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      else
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_label)
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      cdbg << "REDUCE_LEFT" << endl;
      cdbg << "gold_label " << gold_label << endl;
      cdbg << "t_id " << t_id << endl;
    }
    else if (a_id == REDUCE_RIGHT)
    { uint32_t gold_label = 0;
      uint32_t last = stack.last();
      uint32_t second_last = stack[stack.size()-2];
      cdbg << "gold_heads[last].size() " << gold_heads[last].size() << endl;
      cdbg << "gold_heads[last][0] " << gold_heads[last][0] << endl;
      for (size_t j=0; j< gold_heads[last].size(); j++)
        if (gold_heads[last][j] == second_last)
          gold_label = gold_tags[last][j];
      if(gold_label == 0)
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .erase_oracles()
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      else
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_label)
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      cdbg << "REDUCE_RIGHT" << endl;
      cdbg << "gold_label " << gold_label << endl;
      cdbg << "t_id " << t_id << endl;
    }
    else if (a_id == SWAP_REDUCE_RIGHT)
    { uint32_t gold_label = 0;
      uint32_t last = stack.last();
      uint32_t third_last = stack[stack.size()-3];
      for (size_t j=0; j< gold_heads[last].size(); j++)
        if (gold_heads[last][j] == third_last)
          gold_label = gold_tags[last][j];
      if(gold_label == 0)
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .erase_oracles()
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      else
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_label)
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      cdbg << "SWAP_REDUCE_RIGHT" << endl;
      cdbg << "gold_label " << gold_label << endl;
      cdbg << "t_id " << t_id << endl;
    }
    else if (a_id == SWAP_REDUCE_LEFT)
    { uint32_t gold_label = 0;
      uint32_t second_last = stack[stack.size()-2];
      for (size_t j=0; j< gold_heads[second_last].size(); j++)
        if (gold_heads[second_last][j] == idx)
          gold_label = gold_tags[second_last][j];
      if(gold_label == 0)
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .erase_oracles()
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      else
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_label)
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      cdbg << "SWAP_REDUCE_LEFT" << endl;
      cdbg << "gold_label " << gold_label << endl;
      cdbg << "t_id " << t_id << endl;
    }
    else if (a_id == HALLUCINATE)
    { // general psuedocode sketch:
      //   ldf_id = 0
      //   v_array<uint32_t> gold_ids = v_init<uint32_t>();
      //   for each idx in the past the corresponds to a valid concept
      //     example* ldf_ec = *sch.ldf_example(ldf_id);
      //     ldf_set_label(ldf_id, idx);
      //     clear_example_data(*ldf_ec);  // erase whatever is in there
      //     VW::copy_example_data(false, ldf_ec, data->ex);  // copy the current parser state
      //     LabelDict::add_example_namespace_from_memory(concept_to_features, *ldf_ec, idx, 'h');  // put memory features into namespace 'h'
      //     // ^^^----- note, we want to do quadratic between 'h' and any namespace in the normal features
      //     if (this is a correct hallucinate)
      //       gold_ids.push_back(idx);
      //     ldf_id++;
      //   // then we want to make a prediction
      //   pred_id = P.set_input(sch.ldf_example, ldf_id).set_oracle(gold_ids).(blah blah blah).predict()
      // old stuff follows
      gold_ids.erase();
      size_t ldf_id = 0;
      for (uint32_t i=1; i<idx; i++)
      { if (tags[i].size() >  0) //node is already assigned a parent
        { example* ldf_ec = sch.ldf_example(ldf_id);
          cdbg << "(ldf_id, i) " << ldf_id << "," << i << endl;
          if (sch.predictNeedsExample())
          { VW::clear_example_data(*ldf_ec);  // erase whatever is in there
            VW::copy_example_data(false, ldf_ec, data->ex);  // copy the current parser state
            LabelDict::add_example_namespace_from_memory(concept_to_features, *ldf_ec, i, 'h');  // put memory features into namespace 'h'
            // ^^^----- note, we want to do quadratic between 'h' and any namespace in the normal features
          }
          sch.ldf_set_label(ldf_id, i);
          if (get_count_gh(gold_heads[i], stack.last()) > get_count(heads[i], stack.last()))
            gold_ids.push_back(ldf_id);
          ldf_id++;
        }
      }
      cdbg << "ldf_id " << ldf_id << endl;
      cdbg << "gold_ids " << gold_ids << endl;
      cdbg << "stack.last() " << stack.last() << endl;
      action t_id0 = P.set_tag((ptag) count)
              .set_input(sch.ldf_example(), ldf_id)
              .set_oracle(gold_ids)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      cdbg << "t_id0 " << t_id0 << endl;
      t_id = sch.ldf_get_label(t_id0);
      cdbg << "Predicted t_id " << t_id << endl;
      count++;
      idx = transition_hybrid(sch, a_id, idx, t_id);
      a_id = REDUCE_RIGHT;
      uint32_t gold_label = 0;
      cdbg << "gold_heads[t_id] " << gold_heads[t_id] << endl;
      cdbg << "gold_tags[t_id] " << gold_tags[t_id] << endl;

      for (size_t j=0; j<gold_heads[t_id].size(); j++)
        if (gold_heads[t_id][j] == stack[stack.size()-2])
          gold_label = gold_tags[t_id][j];
      cdbg << "gold_label " << gold_label << endl;
      if(gold_label == 0)
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .erase_oracles()
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
      else
      { t_id = P.set_tag((ptag) count)
              .set_input(*(data->ex))
              .set_oracle(gold_label)
              .set_max_allowed(data->amr_num_label)
              .set_condition_range(count-1, sch.get_history_length(), 'p')
              .set_learner_id(a_id)
              .predict();
      }
    }
    count++;

    idx = transition_hybrid(sch, a_id, idx, t_id);
  }
  //only root should be left in the stack at this point
  heads[stack.last()].push_back(0);
  tags[stack.last()].push_back((uint64_t)data->amr_root_label);
  //if (!contains_gh(gold_heads[stack.last()], 0))
  //  sch.loss(1.f);
  float s = smatch_loss(sch,n);
  cdbg << "Setting smatch loss here " << s << endl;
  sch.loss(s);
  if (sch.output().good())
  { for(size_t i=1; i<=n; i++)
    { cdbg << "size for i=" << i << " is " << heads[i].size() << endl;
      if(heads[i].size() == 0)
       sch.output() << "0:0:" << NULL_CONCEPT << endl;
      else
      { sch.output() << heads[i][0]<<":"<<tags[i][0]<<":"<<concepts[i];
        for (size_t j=1; j<heads[i].size();j++)
          sch.output() << ":" << heads[i][j] << ":" << tags[i][j];
        sch.output() <<endl;
      }
    }
    cdbg << "output = " << endl << sch.output().str() << endl;
  }


  if (data->reverse_sentence)
    for (size_t i=0; i<n/2; i++)
    { example* tmp = ec[i];
      ec[i] = ec[n-i-1];
      ec[n-i-1] = tmp;
    }
  
  valid_tags.delete_v();
  gold_ids.delete_v();
}
}
