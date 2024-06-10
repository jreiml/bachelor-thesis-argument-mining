#!/bin/bash

# Ensure enough space
rm -rd models/*

# Strength Regression Model + Results
bash scripts/train/strength_regression_arg_rank_30k/train_all.sh ${1:-42}
bash scripts/predict/strength_regression_arg_rank_30k/evidences_sentences_arg_prob_corr.sh ${1:-42}
bash scripts/predict/strength_regression_arg_rank_30k/mapped_ukp_is_arg_corr.sh ${1:-42}
