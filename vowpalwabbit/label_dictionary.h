namespace LabelDict
{
typedef v_hashmap< size_t, features > label_feature_map;
inline bool size_t_eq(size_t &a, size_t &b) { return (a==b); }

void add_example_namespace(example& ec, char ns, features& fs);
void del_example_namespace(example& ec, char ns, features& fs);

void set_label_features(label_feature_map& lfm, size_t lab, features& fs);
void set_label_features(label_feature_map& lfm, size_t lab, example&  ec);

void add_example_namespaces_from_example(example& target, example& source, char target_namespace=0);
void del_example_namespaces_from_example(example& target, example& source, char target_namespace=0);
void add_example_namespace_from_memory(label_feature_map& lfm, example& ec, size_t lab, char target_namespace='l');
void del_example_namespace_from_memory(label_feature_map& lfm, example& ec, size_t lab, char target_namespace='l');

void free_label_features(label_feature_map& lfm);
}
