/*
  Copyright (c) by respective owners including Yahoo!, Microsoft, and
  individual contributors. All rights reserved.  Released under a BSD (revised)
  license as described in the file LICENSE.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "search.h"
#include "search_generate.h"
#include "cost_sensitive.h"
#include "vw.h"
#include "interactions.h"

namespace CS=COST_SENSITIVE;

namespace GenerateTask {
#define disp(x) #x << '=' << x << ' '
  
Search::search_task task = { "generate", run, initialize, nullptr, nullptr, nullptr};

unsigned int levenshtein_distance(const vector<action>& s1, const vector<action>& s2)   // from WIKIPEDIA
{
  const std::size_t len1 = s1.size(), len2 = s2.size();
  std::vector<unsigned int> col(len2+1), prevCol(len2+1);
  
  for (unsigned int i = 0; i < prevCol.size(); i++)
    prevCol[i] = i;
  for (unsigned int i = 0; i < len1; i++) {
    col[0] = i+1;
    for (unsigned int j = 0; j < len2; j++)
      // note that std::min({arg1, arg2, arg3}) works only in C++11,
      // for C++98 use std::min(std::min(arg1, arg2), arg3)
      col[j+1] = std::min({ prevCol[1 + j] + 1, col[j] + 1, prevCol[j] + (s1[i]==s2[j] ? 0 : 1) });
    col.swap(prevCol);
  }
  return prevCol[len2];
}
  
class IncrementalEditDistance
{
public:
  IncrementalEditDistance(vector<action>& target, action eos=1, bool soft_loss=false, bool verify=false, size_t subst_cost=1, size_t ins_cost=1, size_t del_cost=1)
      : target(target), subst_cost(subst_cost), ins_cost(ins_cost), del_cost(del_cost), N(target.size()), eos(eos), verify(verify), soft_loss(soft_loss)
  { prev_row = new size_t[N+1];
    cur_row  = new size_t[N+1];

    for (size_t n=0; n<=N; n++)
      prev_row[n] = del_cost * n;

    if (soft_loss)
      for (action& a : target)
        target_words[a] ++;

    prev_row_min = 0;
  }

  void append(action a)
  { cur_row[0] = prev_row[0] + ins_cost;
    prev_row_min = cur_row[0];
    for (size_t n=1; n<=N; n++)
    { cur_row[n] = min3( prev_row[n] + ins_cost,
                         prev_row[n-1] + ((target[n-1] == a) ? 0 : subst_cost),
                         cur_row[n-1] + del_cost );
      prev_row_min = min(prev_row_min, cur_row[n]);
    }
    // swap cur_row and prev_row
    size_t* tmp = cur_row;
    cur_row = prev_row;
    prev_row = tmp;
    if (verify)
      verify_prediction.push_back(a);
    if (soft_loss)
      target_words[a] --;
  }

  void append(vector<action>& s) { for (action a : s) append(a); }

  set<action>& next_actions()
  { A.clear();
    for (size_t n=0; n<=N; n++)
    { if (prev_row[n] == prev_row_min)
        A.insert( (n < N) ? target[n] : eos );
    }
    return A;
  }

  void actions_with_cost(v_array<pair<action,float>>& AC)
  { for (auto& p: AC) p.second = 1.;
    if (soft_loss)
      for (auto const& entry : target_words)
        if (entry.second > 0)
          AC[ entry.first-1 ].second = 0.5;
    
    AC[eos-1].second = finish_distance(); // minf(100., (float)(prev_row[N] - prev_row_min)); // </s>
    for (size_t n=0; n<N; n++)
      if (prev_row[n] == prev_row_min)
        AC[ target[n]-1 ].second = 0.;
  }
  
  set<size_t>& next_positions()
  { P.clear();
    for (size_t n=0; n<=N; n++)
    { if (prev_row[n] == prev_row_min)
        P.insert(n);
    }
    return P;
  }

  float minf(float a, float b) { return (a < b) ? a : b; }

  size_t distance() { return prev_row_min; }
  size_t finish_distance()
  { // find last occurance of prev_row_min
    //cerr << "               ["; for (size_t n=0; n<=N; n++) cerr << ' ' << prev_row[n]; cerr << " ]" << endl;
    size_t best = prev_row[N];
    for (size_t n=0; n<N; n++)
      best = min(best, (N-n) * ins_cost + prev_row[n]);

    //int n = (int)N;
    //while (n >= 0 && prev_row[n] > prev_row_min) n--;
    //cerr << "               " << disp(n) << disp(N) << endl;
    //size_t ret = (N-n) * ins_cost + prev_row_min;
    if (verify)
    { size_t dist = levenshtein_distance(target, verify_prediction);
      if (best != dist)
      { cerr << "error: " << best << " != " << dist << endl;
        assert(false);
      }
    }
    return best;
  }

  ~IncrementalEditDistance() { delete[] prev_row; delete[] cur_row; }
  
private:
  size_t* prev_row;
  size_t* cur_row;
  vector<action> target;
  std::map<action,int> target_words;
  vector<action> verify_prediction;
  set<action> A;
  set<size_t> P;
  size_t subst_cost, ins_cost, del_cost, prev_row_min, N;
  action eos;
  bool verify; // should we verify correctness of finish_distance()?
  bool soft_loss;
  
  inline size_t min3(size_t a, size_t b, size_t c) { return (a < b) ? (a < c) ? a : c : (b < c) ? b : c; }
};
    



struct gen_data
{ action K;   // number of output words
  float max_length_ratio;  // max # output = max_length_ratio * # input, default 2
  action eos, oov;  // eos id, default 1; oov default 2
  IncrementalEditDistance* ied; // at training time, for oracle
  vector<size_t> align_out_to_in; // at training time, if alignments are available, align_out_to_in[m] for m in output gives n in input such that n <-> m, or n=0 if none
  size_t max_output_length;
  bool oracle_alignment;
  bool oracle_translation;
  vector<uint32_t> covered;
  vector<string>* en_dict;
  vector<features*> en_features;
  v_array< pair<action,float> >* costs;
  bool action_costs;
  bool remove_oov;
  bool verify_alignment;
  bool soft_loss;
  Search::predictor* P; // cached predictor for speed

  gen_data(size_t _K) :
      K(_K),
      max_length_ratio(1.5),
      eos(1),
      oov(2),
      ied(nullptr),
      max_output_length(40),
      oracle_alignment(true),
      oracle_translation(false),
      costs(nullptr),
      action_costs(false),
      remove_oov(true),
      verify_alignment(false), // need audit
      soft_loss(false),
      P(nullptr)
  {}
};

vector<string>* read_english_dictionary(string fname)
{ ifstream file(fname);
  assert(file.is_open());
  string str, word;
  uint32_t id;
  vector<string>* ret = new vector<string>();
  ret->push_back("(~_~)");
  while(getline(file,str))
  { istringstream ss(str);
    ss >> id;
    ss >> word;
    assert(id == ret->size());
    ret->push_back(word);
  }
  return ret;
}
  
void initialize(Search::search& S, size_t& num_actions, po::variables_map& vm)
{ 
  vw& vw_obj = S.get_vw_pointer_unsafe();
  new_options(vw_obj, "Search Generator Options")
      ("generate_soft_loss", "use soft loss instead of hard edit distance loss")
      ("generate_output_dictionary", po::value<string>(), "dictionary that maps output ids to output words");
  add_options(vw_obj);

  gen_data& G = *new gen_data(num_actions);
  
  G.en_dict = nullptr;
  if (vm.count("generate_output_dictionary") > 0)
    G.en_dict = read_english_dictionary(vm["generate_output_dictionary"].as<string>());
  G.soft_loss = vm.count("generate_soft_loss") > 0;
  
  if ((vw_obj.namespace_dictionaries['e'].size() > 0) && (G.en_dict != nullptr))
  { for (size_t i=0; i<G.en_dict->size(); ++i)
    { string& en = (*G.en_dict)[i];
      char* en_c_str = (char*)en.c_str();
      substring ss = { en_c_str, en_c_str + en.length() };
      // look it up in the feature dictionary
      uint64_t hash = uniform_hash(ss.begin, ss.end-ss.begin, quadratic_constant);
      features* ff = vw_obj.namespace_dictionaries['e'][0]->get(ss, hash);
      G.en_features.push_back(ff);
    }
  }

  if (G.action_costs)
  { G.costs  = new v_array< pair<action,float> >();
    *G.costs = v_init< pair<action,float> >();
    for (action k=1; k<=G.K; k++)
      G.costs->push_back(make_pair(k, 0.));
  }
  
  S.set_task_data<gen_data>(&G);
  S.set_num_learners({true, false}); // LDF for learner 0, not for learner 1
  S.ldf_alloc(G.max_output_length);
  S.set_label_parser( COST_SENSITIVE::cs_label, [](polylabel&l) -> bool { return l.cs.costs.size() == 0; });
  S.set_options(0
                | Search::AUTO_CONDITION_FEATURES
                | Search::NO_CACHING
                | Search::IS_MIXED_LDF
                | G.action_costs * Search::ACTION_COSTS
                );
}


void get_oracle(gen_data& G, vector<example*>& ec)
{ // file format:
  // [output sequence] | <s>
  // | in1
  // | in2
  // | in3=(eg </s> and whatever other features you want)
  // \n
  // where output sequence is something like:
  //    2 3 4 1
  // and "1" means eos (unless otherwise specified by options)
  // if you know some "guesses" at the correct alignment
  // then you can provide this. for instance:
  //    2:1 3:0 4:2 1:3
  // means that "2" (the first word in the output) is aligned to "1"
  // (the first _real_ word in the input, "in1" above), 3 is unaligned
  // (:0 .. meaning aligned to the <s>/null token), 4 is aligned to
  // in2 and </s> is aligned to in3 (which is presumably </s>)
  //
  // for test examples, leave the label on the first line blank

  size_t N = ec.size();
  
  if (G.ied != nullptr)
  { delete G.ied;
    G.ied = nullptr;
  }
  G.align_out_to_in.clear();
  
  if (ec.size() == 0) return;
  v_array<CS::wclass>& lab = ec[0]->l.cs.costs;
  if (lab.size() == 0) return;

  vector<action> target;
  bool has_costs = true;
  for (CS::wclass& wc : lab)
  { target.push_back(wc.class_index);
    if ((wc.x < 0.) || (wc.x >= N))
      has_costs = false;
  }
  target.pop_back();
  //cerr << "target = "; for (auto i : target) cerr << i << ' '; cerr << endl;
  G.ied = new IncrementalEditDistance(target, G.eos, G.soft_loss);
  if (has_costs)
    for (CS::wclass& wc : lab)
      G.align_out_to_in.push_back((size_t) wc.x);

  if (G.verify_alignment)
  { for (size_t i=0; i<target.size(); i++)
    { cerr << (*G.en_dict)[target[i]] << "\t";
      size_t j = G.align_out_to_in[i];
      cerr << j << "\t";
      cerr << ec[j]->feature_space['f'].space_names[0]->first << "/" << ec[j]->feature_space['f'].space_names[0]->second;
      cerr << endl;
    }
  }
}


void reset_learner(Search::predictor& P, size_t learner_id)
{ P.erase_oracles();
  P.erase_alloweds();
  P.set_learner_id(learner_id);
}


set<size_t>* alignment_oracle(gen_data& G)
{ assert(G.ied != nullptr);
  set<size_t>* aln  = new set<size_t>();
  if (G.align_out_to_in.size() > 0)
  { set<size_t>& posn = G.ied->next_positions();
    for (size_t m : posn)
      // we want to produce the mth word in the reference output.
      // need to map it back to a word in the input
      if (m < G.align_out_to_in.size())
        aln->insert(G.align_out_to_in[m]);
  }
  return aln;
}

void inline add_feature(example& ex, uint64_t idx, unsigned char ns, uint64_t mask, uint64_t multiplier, bool audit=false)
{ ex.feature_space[(int)ns].push_back(1.0f, (idx * multiplier) & mask);
  ex.num_features++;
  ex.total_sum_feat_sq += 1.;
}

void add_all_features(example& ex, example& src, unsigned char tgt_ns, uint64_t mask, uint64_t multiplier, uint64_t offset, bool audit=false, float val=1.f)
{ features& tgt_fs = ex.feature_space[tgt_ns];
  for (namespace_index ns : src.indices)
    if(ns != constant_namespace) // ignore constant_namespace
    { for (feature_index i : src.feature_space[ns].indicies)
        tgt_fs.push_back(val, ((i / multiplier + offset) * multiplier) & mask );
      ex.num_features += src.feature_space[ns].indicies.size();
      ex.total_sum_feat_sq += (float)src.feature_space[ns].indicies.size();
    }
}
  
action predict_alignment(Search::search& S, gen_data& G, vector<example*>& ec, size_t m, action last_a, vector<action>& out)
{ size_t N = ec.size();
  Search::predictor& P = *G.P;
  reset_learner(P, 0);
  P.set_tag(2*m+1).set_input(S.ldf_example(), N).set_condition_range(2*m, S.get_history_length(), 'h');

  vw& vw_obj = S.get_vw_pointer_unsafe();
  uint64_t mask = S.get_mask();
  uint64_t multiplier = vw_obj.wpp << vw_obj.reg.stride_shift;
  
  set<size_t>* oracle = nullptr;
  if ((G.ied != nullptr) && (true || S.predictNeedsReference()))
  { oracle = alignment_oracle(G);
    //cerr << "oracle = {"; for (auto i : *oracle) cerr << ' ' << i; cerr << " }" << endl;
  }

  if (G.oracle_alignment)
  { assert(oracle != nullptr);
    action ret = 0;
    for (auto a : *oracle) { ret = a; break; }
    delete oracle;
    return ret;
  }

  for (size_t a=0; a<N; a++)
  { // for each word in the input, create an LDF example that corresponds to "align here!"
    if (true || S.predictNeedsExample())
    { example& ex = * S.ldf_example(a);
      // features are based on ec[last_a] and ec[a], plus last_a vs a
      VW::clear_example_data(ex);
      VW::copy_example_data(false, &ex, ec[a]);
      ex.weight = 1.;
      ex.indices.push_back((size_t)'p');
      add_all_features(ex, *ec[last_a], 'p', mask, multiplier, 4390197);
      add_feature(ex, multiplier * (4319041 + ((int)a - (int)last_a)), 'p', mask, multiplier);
      add_feature(ex, multiplier * (8902137 + (a > last_a)),           'p', mask, multiplier);
      add_feature(ex, multiplier * (9403183 + (G.covered[a])),         'p', mask, multiplier);
      add_feature(ex, multiplier * (8590137 + ((int)N - (int)out.size())),        'p', mask, multiplier);
    }
    S.ldf_set_label(a, a+1, 0.);

    if ((oracle != nullptr) && (oracle->find(a) != oracle->end()))
      P.add_oracle(a);
  }
  if (oracle != nullptr) delete oracle;
  
  action a = P.predict();
  assert(a >= 0);
  assert(a <  N);
  return a;
}

action predict_word(Search::search& S, gen_data& G, vector<example*>& ec, size_t m, action last_a, action a, vector<action>& out)
{
  if (G.oracle_translation)
  { for (action w : G.ied->next_actions())
      return w;
    assert(false);
  }
  //cerr << "       out = "; for (auto i : out) cerr << i << ' '; cerr << endl;
  //cerr << "    oracle = {"; for (auto i : G.ied->next_actions()) cerr << ' ' << i; cerr << " }" << endl;

  vw& vw_obj = S.get_vw_pointer_unsafe();
  uint64_t mask = S.get_mask();
  uint64_t multiplier = vw_obj.wpp << vw_obj.reg.stride_shift;
  
  size_t N = ec.size();
  Search::predictor& P = *G.P;
  example& ex = * S.ldf_example(0);
  reset_learner(P, 1);
  P.set_tag(2*m+2).set_input(ex).set_condition_range(2*m+1, S.get_history_length(), 'h');
  // features of word being translated
  VW::clear_example_data(ex);
  VW::copy_example_data(false, &ex, ec[a]);
  ex.weight = 1.;
  ex.indices.push_back((size_t)'l');  // left fr context
  ex.indices.push_back((size_t)'j');  // more left fr context
  ex.indices.push_back((size_t)'r');  // right fr context
  ex.indices.push_back((size_t)'s');  // more right fr context
  ex.indices.push_back((size_t)'p');  // alignment features
  ex.indices.push_back((size_t)'q');  // previous alignment fr context
  ex.indices.push_back((size_t)'b');  // bag of words context
  ex.indices.push_back((size_t)'e');  // previous english output context
  // features of neighboring words
  if (a > 0)   add_all_features(ex, *ec[a-1], 'l', mask, multiplier, 48931043, false, 0.5);
  else         add_feature(ex, multiplier * (48931043 + 483910741), 'l', mask, multiplier);
  if (a > 1)   add_all_features(ex, *ec[a-2], 'j', mask, multiplier, 148931043, false, 0.5);
  else         add_feature(ex, multiplier * (148931043 + 483910741), 'j', mask, multiplier);
  if (a < N-1) add_all_features(ex, *ec[a+1], 'r', mask, multiplier, 9831487, false, 0.5);
  else         add_feature(ex, multiplier * (9831487 + 483910741), 'l', mask, multiplier);
  if (a < N-2) add_all_features(ex, *ec[a+2], 's', mask, multiplier, 19831487, false, 0.5);
  else         add_feature(ex, multiplier * (19831487 + 483910741), 's', mask, multiplier);
  for (size_t n=0; n<N; n++)
    add_all_features(ex, *ec[n], 'b', mask, multiplier, 3489101, false, 0.5 /* * exp(0. - G.covered[n]) */ / (float)N);
  // features of previously translated word
  add_all_features(ex, *ec[last_a], 'q', mask, multiplier, 94031871);
  add_feature(ex, multiplier * (4319041 + ((int)a - (int)last_a)), 'p', mask, multiplier);
  add_feature(ex, multiplier * (8902137 + (a > last_a)),           'p', mask, multiplier);
  add_feature(ex, multiplier * (9403183 + (G.covered[a])),         'p', mask, multiplier);
  add_feature(ex, multiplier * (8590137 + ((int)N - (int)out.size())),        'p', mask, multiplier);

  // previous english context
  size_t M = out.size();
  for (size_t delta=1; delta<=3; delta++)
  { int m = (int)M-(int)delta;
    if (m < 0)
      add_feature(ex, multiplier * (84930177 + 4983107 * delta), 'e', mask, multiplier);
    else 
    { // there in an output word at m
      if (out[m] < G.en_features.size())
      { features* ff = G.en_features[out[m]];
        if (ff != nullptr)
          for (size_t i=0; i<ff->indicies.size(); i++)
            add_feature(ex, multiplier * (84930177 + 493107 * delta * (49101 * ff->indicies[i] /* / multiplier */)), 'e', mask, multiplier);
      }
      // just vanilla feature without dictionary
      add_feature(ex, multiplier * (84930177 + 4983107 * delta * (49101 * out[m] + 840178103)), 'e', mask, multiplier);
    }
  }
  
  // cheating features:
  //for (action w : G.ied->next_actions())
  //  add_feature(ex, multiplier * (4890137 + w), 'p', mask, multiplier);
  

  size_t new_count = 0;
  float new_weight = 0.f;
  INTERACTIONS::eval_count_of_generated_ft(vw_obj, ex, new_count, new_weight);
  ex.num_features += new_count;
  ex.total_sum_feat_sq += new_weight;
  
  if ((G.ied != nullptr) && (true || S.predictNeedsReference()))
  { if (G.action_costs)
    { G.ied->actions_with_cost(* G.costs);
      P.set_allowed(*G.costs);
    }
    else
      for (action w : G.ied->next_actions())
        P.add_oracle(w);
  }
  return P.predict();
}

void print_word(gen_data& G, std::stringstream& out, action w)
{ if (G.en_dict == nullptr)
    out << w << ' ';
  else if (w < G.en_dict->size())
    out << (*G.en_dict)[w] << ' ';
  else
    out << "*OOV* ";
}
  
void run(Search::search& S, vector<example*>& ec)
{
  gen_data& G = *S.get_task_data<gen_data>();
  G.P = new Search::predictor(S, (ptag)0);

  size_t N = ec.size();
  size_t M = max(5, min(G.max_output_length, (size_t)( G.max_length_ratio * N )));
  action last_a = 0;   // assume that example[0] is padding <s>
  vector<action> out;
  get_oracle(G, ec);

  G.covered.clear();
  for (size_t n=0; n<N; n++) G.covered.push_back(0);

  //cerr << disp(N) << disp(M) << endl;
  for (size_t m=0; m<M; m++)
  {
    // we need to produce the mth word, so first we need to pick a
    // place to align to
    action a = predict_alignment(S, G, ec, m, last_a, out);
    //cerr << disp(m) << disp(a) << endl;
        
    // now, predict the word id
    action w = predict_word(S, G, ec, m, last_a, a, out);

    //cerr << disp(w) << endl;
        
    // check to see if we're done
    if (w == G.eos)
      break;
    
    // update internal state
    if (G.ied != nullptr)
      G.ied->append(w);
    
    out.push_back(w);
    
    G.covered[a]++;
    last_a = a;

    if (S.output().good())
      print_word(G, S.output(), w);
    //S.output() << w << ':' << a << ' ';
  }
  //if (S.output().good()) S.output() << N << '_' << M << '_' << out.size();
  if (G.ied != nullptr)
  { size_t loss = G.ied->finish_distance();
    S.loss((float)loss);
    //if (S.output().good()) S.output() << loss;
    //cerr << "       out = "; for (auto i : out) cerr << i << ' '; cerr << endl;
    //cerr << "      loss = " << loss << endl;
    delete G.ied;
    G.ied = nullptr;
  }
  delete G.P;
}

}
