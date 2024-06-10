#!/bin/bash

# Ensure enough space
rm -rd models/*

# Argument Detection Model + Results
bash scripts/train/argument_detection_ukp/train_all.sh ${1:-42}
bash scripts/predict/argument_detection_ukp/alpha_mapped_arg_rank_30k_strength_prob_corr.sh ${1:-42}
bash scripts/predict/argument_detection_ukp/alpha_mapped_evidences_sentences_arg_prob_corr.sh ${1:-42}
bash scripts/predict/argument_detection_ukp/regression_mapped_arg_rank_30k_strength_prob_corr.sh ${1:-42}
bash scripts/predict/argument_detection_ukp/regression_mapped_evidences_sentences_arg_prob_corr.sh ${1:-42}
