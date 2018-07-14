#include <stdio.h>
#include <stdlib.h> // for system
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../vowpalwabbit/vw.h"
#include "../vowpalwabbit/ezexample.h"
#include "libsearch.h"

size_t sed(const string &s1, const string &s2, size_t subst_cost=1, size_t ins_cost=1, size_t del_cost=1);

#define myhash(a,b) 93187*(3891*a + 89307*b)

char _action2char[256];
action _char2action[256];

int EOS=1, OOV=2, SKIP=3, COPY=4, N_SPECIAL=5;
action n_actions;

void setup_character_tables(string& charlist)
{ for (int i=0; i<256; i++)
    { _action2char[i] = 0;
      _char2action[i] = OOV;
    }
  for (int i=0; i<min(255, charlist.length()); i++)
    { _action2char[i+N_SPECIAL] = charlist[i];
      _char2action[charlist[i]] = i+N_SPECIAL;
    }
  n_actions = N_SPECIAL + charlist.length();
}

action char2action(char c)
{ if (c < N_SPECIAL) return c;
  return _char2action[c];
}

char action2char(action i)
{ if (i > n_actions) throw exception();
  if (i < N_SPECIAL) return i;
  return _action2char[i];
}

struct nextstr
{ char c;
  float cw;
  string s;
  float sw;
  nextstr(char _c, float _cw, string _s, float _sw) : c(_c), cw(_cw), s(_s), sw(_sw) {}
};

class notstring
{
public:
  notstring() { reset((uint32_t)' '); }
  notstring(char c) { reset((uint32_t)c); }

  void reset(uint32_t c) { d = 48390131; d = append(c); }
  
  notstring& operator+=(char c) { d = append((uint32_t)c); return *this; }
  uint32_t get(char c) { return append((uint32_t)c); }

  uint32_t append(uint32_t i) { return 34810347 * (d + i * 3489101); }
  
private:
  uint32_t d;
};

class Trie
{
public:
  Trie() : terminus(false), count(0), max_count(0), max_string("") {}

  ~Trie()
  { for (Trie* t : children)
      delete t;
  }

  Trie* step(const char c)
  { size_t id = char2action(c) - 1;
    if (children.size() <= id) return nullptr;
    return children[id];
  }

  void insert(const char*str, size_t c=1)
  { if (str == nullptr || *str == 0)
    { terminus += c;
      count    += c;
    }
    else
    { count += c;
      size_t id = char2action(*str) - 1;
      while (children.size() <= id)
        children.push_back(nullptr);
      if (children[id] == nullptr)
        children[id] = new Trie();
      children[id]->insert(str+1, c);
    }
  }

  size_t contains(const char*str)
  { if (str == nullptr || *str == 0)
      return terminus;
    size_t id = char2action(*str) - 1;
    if (children.size() <= id) return 0;
    if (children[id] == nullptr) return 0;
    return children[id]->contains(str+1);
  }

  void get_next(const char*prefix, vector<nextstr>& next)
  { if (prefix == nullptr || *prefix == 0)
    { next.clear();
      float c = 1. / (float)count;
      next.push_back( nextstr(EOS, log(1. + c * (float)terminus), max_string, log(1. + (float)max_count)) );
      for (size_t id=0; id<children.size(); id++)
        if (children[id])
          next.push_back( nextstr(action2char(id+1), c*(float)children[id]->count, children[id]->max_string, log(1.+ (float)children[id]->max_count)) );
    }
    else
    { size_t id = char2action(*prefix) - 1;
      if (children.size() <= id) return;
      if (children[id] == nullptr) return;
      children[id]->get_next(prefix+1, next);
    }
  }

  void build_max(string prefix="")
  { max_count = terminus;
    max_string = prefix;
    for (size_t id=0; id<children.size(); id++)
      if (children[id])
      { char c = action2char(id + 1);
        children[id]->build_max(prefix + c);
        if (children[id]->max_count > max_count)
        { max_count  = children[id]->max_count;
          max_string = children[id]->max_string;
        }
      }
  }

  void print(char c='^', size_t indent=0)
  { cerr << string(indent*2, ' ');
    cerr << '\'' << c << "' " << count << " [max_string=" << max_string << " max_count=" << max_count << "]" << endl;
    for (size_t i=0; i<children.size(); i++)
      if (children[i])
        children[i]->print(action2char(i+1), indent+1);
  }

    
  
private:
  size_t terminus;   // count of words that end here?
  size_t count;      // count of all words under here (including us)
  size_t max_count;  // count of most frequent word under here
  string max_string; // the corresponding string
  vector<Trie*> children;
};

class IncrementalEditDistance
{
public:
  IncrementalEditDistance(string& target, size_t subst_cost=1, size_t ins_cost=1, size_t del_cost=1)
    : target(target), subst_cost(subst_cost), ins_cost(ins_cost), del_cost(del_cost), N(target.length()), output_string("")
  { prev_row = new size_t[N+1];
    cur_row  = new size_t[N+1];

    for (size_t n=0; n<=N; n++)
      prev_row[n] = del_cost * n;

    prev_row_min = 0;
  }

  void append(char c)
  { output_string += c;
    cur_row[0] = prev_row[0] + ins_cost;
    prev_row_min = cur_row[0];
    for (size_t n=1; n<=N; n++)
    { cur_row[n] = min3( prev_row[n] + ins_cost,
                         prev_row[n-1] + ((target[n-1] == c) ? 0 : subst_cost),
                         cur_row[n-1] + del_cost );
      prev_row_min = min(prev_row_min, cur_row[n]);
    }
    // swap cur_row and prev_row
    size_t* tmp = cur_row;
    cur_row = prev_row;
    prev_row = tmp;
  }

  void append(string s) { for (char c : s) append(c); }

  vector<char>& next()
  { A.clear();
    for (size_t n=0; n<=N; n++)
    { if (prev_row[n] == prev_row_min)
        A.push_back( (n < N) ? target[n] : EOS );
    }
    return A;
  }

  float minf(float a, float b) { return (a < b) ? a : b; }

  vector< pair<action,float> > all_next(unsigned int input_pos, string& in_in)
  { vector< pair<action,float> > B;
    for (size_t a=1; a<=n_actions; a++)
      B.push_back( make_pair(a, 1.) );
    B[ char2action(EOS)-1 ].second = minf(100., (float)(prev_row[N] - prev_row_min));
    for (size_t n=0; n<N; n++)
      if (prev_row[n] == prev_row_min)
	//{ cerr << n << "," << target[n] << ' '; 
	  B[ char2action(target[n])-1 ].second = 0.; 
    //cerr << "{" << in_in[input_pos] << ":" << char2action(in_in[input_pos]) << "}";
    B[ char2action(COPY)-1 ].second = B[ char2action(in_in[input_pos])-1 ].second;
    int skip = char2action(SKIP)-1; // skip
    if (input_pos+1 < in_in.length()) B[skip].second = min(B[skip].second, B[char2action(in_in[input_pos+1])-1].second+0.2);
    if (input_pos+2 < in_in.length()) B[skip].second = min(B[skip].second, B[char2action(in_in[input_pos+2])-1].second+0.5);
    //cerr << "costs = <"; for (size_t a=0; a<n_actions; a++) cerr << " " << (int)action2char(B[a].first) << "," << B[a].second; cerr << " >"; cerr
    return B;
  }

  string out() { return output_string; }
  size_t distance() { return prev_row_min; }
  size_t finish_distance()
  { // find last occurance of prev_row_min
    int n = (int)N;
    while (n >= 0 && prev_row[n] > prev_row_min) n--;
    return (N-n) * ins_cost + prev_row_min;
  }

  ~IncrementalEditDistance() { delete[] prev_row; delete[] cur_row; }

private:
  size_t* prev_row;
  size_t* cur_row;
  string  target;
  size_t  subst_cost, ins_cost, del_cost, prev_row_min, N;
  string  output_string;
  vector<char> A;

  inline size_t min3(size_t a, size_t b, size_t c) { return (a < b) ? (a < c) ? a : c : (b < c) ? b : c; }
};

struct input
{ string in;
  string out;
  float weight;
  input(string _in, string _out, float _weight) : in(_in), out(_out), weight(_weight) {}
  input(string _in, string _out) : in(_in), out(_out), weight(1.) {}
  input(string _in) : in(_in), out(_in), weight(1.) {}
};

typedef string output;

#define minf(a,b) (((a) < (b)) ? (a) : (b))

float max_cost = 100.;

float get_or_one(vector< pair<char,size_t> >& v, char c)
{ // TODO: could binary search
  for (auto& p : v)
    if (p.first == c)
      return minf(max_cost, (float)p.second);
  return 1.;
}

class Generator : public SearchTask<input, output>
{
public:

  Generator(vw& vw_obj, Trie* _dict=nullptr) : SearchTask<input,output>(vw_obj), dist(0), dict(_dict)    // must run parent constructor!
  { sch.set_options( Search::AUTO_CONDITION_FEATURES | Search::NO_CACHING | Search::ACTION_COSTS );  // TODO: if action costs is specified but no allowed actions provided, don't segfault :P
    HookTask::task_data& d = *sch.get_task_data<HookTask::task_data>();
    if (d.num_actions != n_actions) throw exception();
  }

  void _run(Search::search& sch, input& in, output& out)
  { IncrementalEditDistance ied(in.out);

    Trie* cdict = dict;

    int remaining_char_count[256];
    for (size_t i=0; i<256; i++) remaining_char_count[i] = 0;
    
    //cerr << "--------------" << endl;
    v_array<action> ref = v_init<action>();
    int N = in.in.length();

    for (size_t n=0; n<in.in.length(); n++)
      remaining_char_count[in.in[n]]++;
    
    out = "";
    vector<nextstr> next;
    unsigned int input_pos=0;
    for (int m=1; m<=N*1.2; m++)     // at most |in|*1.2 outputs
    { ezexample ex(&vw_obj);

      // length info
      ex(vw_namespace('l'))
	(myhash('l', 'N'), (float)N)
	(myhash('l', 'm'), (float)m);
      if (N != m)
        ex(myhash('l', 'd'), (float)(N-m));

      // suffixes thus far
      ex(vw_namespace('s'));
      notstring tmp('s');
      for (int i=out.length(); i >= m-15 && i >= 0; i--)
	{ tmp += out[i]; // tmp = out[i] + tmp;
	  ex(tmp.get('p'));
	  //ex("p=" + tmp);
      }

      // characters thus far
      ex(vw_namespace('c'));
      ex(myhash('c', 1));
      for (char c: out) ex(myhash('c', c));
      ex(myhash('c', 2));
      //ex("c=^");
      //for (char c : out) ex("c=" + string(1,c));
      //ex("c=$");

      // words thus far
      ex(vw_namespace('w'));
      tmp.reset('w');
      for (char c : out)
      { if (c == '^') continue;
        if (c == ' ')
	  { ex(tmp.get('w'));
	    tmp.reset('w');
	    //ex("w=" + tmp + "$");
	    //tmp = "";
	  }
        else tmp += c;
      }
      ex(tmp.get('w')); 

      // do we match the trie?
      if (cdict)
      { next.clear();
        cdict->get_next(nullptr, next);
        ex(vw_namespace('d'));
        char best_char = '~'; float best_count = 0.;
        for (auto xx : next)
	  { if (xx.cw > 0.) ex(myhash('d', myhash('c', xx.c)), xx.cw);
	    if (xx.sw > 0.) ex("mc=" + xx.s, xx.sw);
	    if (xx.sw > best_count) { best_count = xx.sw; best_char = xx.c; }
	  }
        if (best_count > 0.)
          ex(myhash('d', myhash('b', best_char)), best_count);
      }

      // input characters
      ex(vw_namespace('C'));
      ex(myhash('C', 1));
      for (int n=0; n<N; n++)
	ex(myhash('C', in.in[n]));
      ex(myhash('C', 2));
      /*
      ex("c=^");
      for (int n=0; n<N; n++)
        ex("c=" + in.in[n]);
      ex("c=$");
      */

      // input words
      ex(vw_namespace('W'));
      tmp.reset('W');
      for (char c : in.in)
      { if (c == ' ')
	  { ex(tmp.get('w'));
	    tmp.reset('W');
	  }
        else tmp += c;
      }

      /*
      ex(tmp.get('w'));
      
      // input focus
      ex(vw_namespace('f'));
      tmp.reset('f');
      for (int delta=1; delta<=5; delta++)
	{ int d = (delta / 2);
	  if (delta % 2 == 1) d = -d;
	  int j = input_pos+d;
	  char c = (j < 0) ? '^' : (j >= N) ? '$' : in.in[j];
	  tmp += c;
	  ex(tmp.get('p'));
	}
      
      ref.erase();
      */
      ex("w=" + tmp);
      ref.clear();

      // remaining character count
      ex(vw_namespace('R'));
      for (size_t i=0; i<255; i++)
	if (remaining_char_count[i] > 0)
	  ex(myhash('R', myhash(i, remaining_char_count[i])));
      

      /*
      vector<char>& best = ied.next();
      if (best.size() == 0) ref.push_back( char2action('$') );
      else for (char c : best) ref.push_back( char2action(c) );
      char c = action2char( Search::predictor(sch, m)
                            .set_input(* ex.get())
                            .set_oracle(ref)
                            .predict() );
      */

      //cerr << "'" << out << "' ";
      vector< pair<action,float> > all = ied.all_next(input_pos, in.in);
      bool copy_ok = false;
      for (auto p : all)
	if (p.second == 0)
	  { ref.push_back(p.first);
	    if (p.first == COPY) copy_ok = true;
	  }
      if (copy_ok)
	{ ref.erase();
	  ref.push_back(COPY);
	}
      action act = Search::predictor(sch, m)
                            .set_input(* ex.get())
                            .set_allowed(all)
	// 	                    .set_oracle(ref)
                            .predict();
      //cerr << " (" << act << " -> " << (int)action2char(act) << ")" << endl;
      char c = action2char( act );
      if (c == EOS) break;
      if (c == OOV) c = '?';
      if (c == COPY) c = in.in[input_pos];
      if (c != SKIP) // insert
	{ out += c;
	  ied.append(c);
	  if (cdict) cdict = cdict->step(c);
	  if (remaining_char_count[c] > 0)
	    remaining_char_count[c] --;
	}
      input_pos = min(input_pos+1, N-1);
    }

    dist = ied.finish_distance();
    sch.loss((float)dist * in.weight);
  }

  size_t get_dist() { return dist; }

private:
  size_t dist;
  Trie* dict;
};

class Counter
{
public:
  Counter() : num(0.), den(0.) {}
  void add(float multiplier, float weight=1.0)
  { den += weight;
    num += multiplier * weight;
  }
  void reset() { num = 0.; den = 0.; }
  float get() { return num / max(den, 1.0); }
  
private:
  float num, den;
};

void run_easy()
{ vw& vw_obj = *VW::initialize("--search 31 --quiet --search_task hook --ring_size 1024 --search_rollin learn --search_rollout none");
  Generator task(vw_obj);
  output out("");

  vector<input> training_data =
  { input("maison", "house"),
    input("lune", "moon"),
    input("petite lune", "little moon"),
    input("la fleur", "the flower"),
    input("petite maison", "little house"),
    input("fleur", "flower"),
    input("la maison", "the house"),
    input("grande lune", "big moon"),
    input("grande fleur", "big flower")
  };
  vector<input> test_data =
  { input("petite fleur", "little flower"),
    input("grande maison", "big house")
  };
  for (size_t i=0; i<100; i++)
  { //if (i == 9999) max_cost = 1.;
    if (i % 10 == 0) cerr << '.';
    for (auto x : training_data)
      task.learn(x, out);
  }
  cerr << endl;

  for (auto x : training_data)
  { task.predict(x, out);
    cerr << "output = " << out << endl;
  }
  for (auto x : test_data)
  { task.predict(x, out);
    cerr << "output = " << out << endl;
  }
}

Trie load_dictionary(const char* fname)
{ ifstream h(fname);
  Trie t;
  string line;
  while (getline(h,line))
  { const char* str = line.c_str();
    char* space = (char*)strchr(str, ' ');
    if (space)
    { *space = 0;
      space++;
      t.insert(space, atof(str));
    }
    else
      t.insert(str);
  }
  return t;
}

void run_istream(Generator& gen, const char* fname, bool is_learn=true, size_t print_every=0)
{ ifstream h(fname);
  if (! h.is_open())
  { cerr << "cannot open file " << fname << endl;
    throw exception();
  }
  string line;
  output out;
  size_t n = 0;

  Counter dist, exact, dist_diff, exact_diff;
  while (getline(h, line))
  { n++;
    //if (n % 500 == 0) cerr << '.';
    size_t i = line.find(" ||| ");
    size_t j = line.find(" ||| ", i+1);
    size_t k = line.find(" ||| ", j+1);
    if (i == string::npos || j == string::npos || k == string::npos)
    { cerr << "skipping line " << n << ": '" << line << "'" << endl;
      continue;
    }
    input dat(line.substr(i+5,j-i-5), line.substr(j+5,k-j-5), atof(line.substr(0,i).c_str())/10.);
    //cerr << "count=" << dat.weight << ", in='" << dat.in << "', out='" << dat.out << "'" << endl;
    //weight += dat.weight;

    if ((not is_learn) || ((print_every>0) && (n % print_every == 0) || (n % 10 == 0)))
      {
	gen.predict(dat, out);
	int this_dist = gen.get_dist();
	assert((this_dist > 0) == (out != dat.out));

	dist.add((float)this_dist, dat.weight);
	exact.add((this_dist == 0) ? 1.0 : 0.0, dat.weight);
	bool same = true;
	if (dat.in != dat.out)
	  { dist_diff.add((float)this_dist, dat.weight);
	    exact_diff.add((this_dist == 0) ? 1.0 : 0.0, dat.weight);
	    same = false;
	  }
    
	//dist += dat.weight * (float)this_dist;
	//if (this_dist == 0) exact += dat.weight;
	if (print_every>0 && (n % print_every == 0))
	  cerr << n << "\t dist=" << dist.get() << " exact=" << exact.get() << " [diff dist=" << dist_diff.get() << " exact=" << exact_diff.get() << " ||| ex same=" << same << " dist=" << this_dist << " input=^" << dat.in << "$  truth=^" << dat.out << "$ pred=^" << out << "$" << endl;
      }
    
    if (is_learn)
      gen.learn(dat, out);
  }
  //if (n > 500) cerr << endl;
  if (!is_learn)
    { cerr << "AVERAGE DISTANCE: " << dist.get() << " [on diff only: " << dist_diff.get() << "]" << endl;
      cerr << "     EXACT MATCH: " << exact.get() << " [on diff only: " << exact_diff.get() << "]" << endl;
    }
}

void train()
{ // initialize VW as usual, but use 'hook' as the search_task
  Trie dict = load_dictionary("speller/short.vocab");
  dict.build_max();
  //dict.print();

  //namespaces:
  //  l: length info
  //  s: output suffixes
  //  c: output characters
  //  w: output words
  //  d: output trie
  //  C: input characters
  //  W: input words
  //  f: input focus
  //  R: remaining characters
  string init_str("--search " + std::to_string(n_actions) + " -l 0.1 -b 29 --quiet --search_task hook --ring_size 1024 --search_rollin ref --search_rollout none -qi: --ngram c5 --skips c2 --ngram w2 --ngram C5 --skips c2 --ngram W2"); //  --search_use_passthrough_repr"); // -q si -q wi -q ci -q di  -f my_model
  vw& vw_obj = *VW::initialize(init_str);
  cerr << init_str << endl;
  Generator gen(vw_obj, nullptr); // &dict);
  for (size_t pass=1; pass<=1; pass++)
  { cerr << "===== pass " << pass << " =====" << endl;
    run_istream(gen, "speller/short.tr", true, 10000);
    //run_istream(gen, "speller/short.tr", false, 1); //300000);
    run_istream(gen, "speller/short.te", false, 10000); //100000);
  }
  VW::finish(vw_obj);
}

void predict()
{ vw& vw_obj = *VW::initialize("--quiet -t --ring_size 1024 -i my_model");
  //run(vw_obj);
  VW::finish(vw_obj);
}

int main(int argc, char *argv[])
{ /*
  string target(argv[1]);
  cerr << "target = " << target << endl;
  IncrementalEditDistance ied(target);
  cerr << "^: ";
  for (size_t i=0; i<=strlen(argv[2]); i++) {
    vector< pair<action,float> > next = ied.all_next();
    for (auto& p : next)
      cerr << action2char(p.first) << ' ' << p.second << "\t";
    cerr << endl;
    cerr << argv[2][i] << ": ";
    ied.append(argv[2][i]);
  }
  cerr << endl;
  */
  /*
  string target("abcde");
  IncrementalEditDistance ied(target);
  ied.append(string("cde"));
  while (true) {
    vector<char>& best = ied.next();
    cerr << ied.out() << " / " << ied.distance() << " -> "; for (char c : best) cerr << c; cerr << endl;
    char c = best[0];
    if (c == '$') break;
    ied.append(c);
  }
  cerr << "final: " << ied.distance() << "\t" << ied.out() << endl;
  return 0;
  */
  string charlist("abcdefghijklmnopqrstuvwxyz0123456789 ");
  if (argc > 1)
    charlist = string(argv[1]) + " ";
  cerr << "charlist = [" << charlist << "]" << endl;
  setup_character_tables(charlist);
  cerr << "n_actions = " << n_actions << endl;
  train();
  //predict();
  //run_easy();
}

/**
with strings

$ time ./search_generate
charlist = [abcdefghijklmnopqrstuvwxyz0123456789 ]
n_actions = 42
--search 42 -l 0.1 -b 29 --quiet --search_task hook --ring_size 1024 --search_rollin ref --search_rollout none -qi: --ngram c5 --skips c2 --ngram w2 --ngram C5 --skips c2 --ngram W2
===== pass 1 =====
10000    dist=2.58789 exact=0.0424242 [diff dist=2.70544 exact=0.00543478 ||| ex same=0 dist=2 input=^when do marajuna plants bud$  truth=^when do marijuana plants bud$ pred=^when do marajuna plants bud$
10000    dist=2.06898 exact=0.0619616 [diff dist=2.20361 exact=0.00925925 ||| ex same=0 dist=2 input=^ie skating animations$  truth=^ice skating animations$ pred=^ie skating animation$
AVERAGE DISTANCE: 2.06898 [on diff only: 2.20361]
     EXACT MATCH: 0.0619616 [on diff only: 0.00925925]

real    0m58.902s
user    0m50.906s
sys     0m6.734s


without strings

**/
