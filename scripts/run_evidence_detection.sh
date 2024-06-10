#!/bin/bash

# Ensure enough space
rm -rd models/*

# Argument Evidence Model + Results
bash scripts/train/argument_probability_regression_evidences_sentences/train_all.sh ${1:-42}
bash scripts/predict/argument_probability_regression_evidences_sentences/arg_rank_30k_strength_prob_corr.sh ${1:-42}
bash scripts/predict/argument_probability_regression_evidences_sentences/mapped_ukp_is_arg_corr.sh ${1:-42}
