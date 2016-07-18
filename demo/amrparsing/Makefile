SHELL=/bin/bash
VW=/fs/clip-amr/vowpal_wabbit/vowpalwabbit/vw

data_dir=/fs/clip-amr/vowpal_wabbit/demo/amrparsing/data
#train_data=$(data_dir)/amr-release-1.0-training-full/amr-release-1.0-training-full
#train_data=$(data_dir)/amr-release-1.0-training-dev-full/amr-release-1.0-training-dev-full
#train_data=$(data_dir)/amr-release-1.0-training-dev-full/amr-release-1.0-training-dev-full.withdep
#test_data=$(data_dir)/amr-release-1.0-test-full/amr-release-1.0-test-full
#test_data=$(data_dir)/amr-release-1.0-test-full/amr-release-1.0-test-full.withdep
concepts_dict=$(data_dir)/amr-release-1.0-training-full/concepts.p
relations_dict=$(data_dir)/amr-release-1.0-training-full/relations.p
span_concepts_dict=$(data_dir)/amr-release-1.0-training-full/span_concept_dict
concept_graph_fragment_dict=$(data_dir)/amr-release-1.0-training-full/concept_graph_fragment_dict.p
scripts_dir=/fs/clip-amr/vowpal_wabbit/demo/amrparsing
smatch_dir=/fs/clip-amr/smatch_2.0

ifeq ($(WITHDEP), True)
  train_data=$(data_dir)/amr-release-1.0-training-dev-full/amr-release-1.0-training-dev-full.withdep
  test_data=$(data_dir)/amr-release-1.0-test-full/amr-release-1.0-test-full.withdep
else
  train_data=$(data_dir)/amr-release-1.0-training-dev-full/amr-release-1.0-training-dev-full
  test_data=$(data_dir)/amr-release-1.0-test-full/amr-release-1.0-test-full
endif

ifeq ($(INPASS),True)
  INPASS_ARG=--inpass
endif

ifeq ($(MULTITASK),True)
  MULTITASK_ARG=--multitask
endif

ifeq ($(NN_UNITS),)
  NN_UNITS_ARG=0
else
  NN_UNITS_ARG=$(NN_UNITS)
endif
.SECONDARY:

all:
	@cat README.md
clean:
	rm -f $(train_data).model $(test_data).pred $(train_data).*.cache  $(train_data)*.writing

amr.model: $(train_data).vw
	@$(VW) --search 118 --search_task amr_parser --amr_num_label 118 --amr_dictionary $(span_concepts_dict) -d $(train_data).vw --search_rollin $(ROLLIN) --search_rollout $(ROLLOUT) -k --cache_file $(train_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).cache --passes $(PASSES) --holdout_after 10312  --search_no_caching -b 25 --search_history_length 3  -f $(train_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).model --ftrl --nn $(NN_UNITS_ARG) $(INPASS_ARG) $(MULTITASK_ARG) --amr_max_hallucinate 0 --classificationesque --csoaa_ldf mc --search_beta 0.5 --search_alpha 1e-5 --search_constant_beta --loss_function hinge

amr.test.predictions: $(test_data).vw amr.model
	@$(VW) -d $(test_data).vw -t --amr_dictionary $(span_concepts_dict) -i $(train_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).model -p $(test_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).pred --amr_max_hallucinate 0
 
amr.test.parse: amr.test.predictions 
	python $(scripts_dir)/vw_pred_to_amr.py $(test_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).pred $(concepts_dict) $(relations_dict) $(concept_graph_fragment_dict) $(test_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).pred.amr

amr.perf: amr.test.parse
	python $(smatch_dir)/smatch.py -f $(test_data).aligned $(test_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).pred.amr --pr
	rm $(train_data).$(PASSES).$(INPASS).$(MULTITASK).$(NN_UNITS).$(ROLLIN).$(ROLLOUT).cache

.PHONY: all clean
