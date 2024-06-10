#!/bin/bash

# In-Topic
## WA
### No Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-in_topic_ukp_is_arg_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-in_topic/checkpoint-*

### Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-in_topic_ukp_is_arg_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-in_topic/checkpoint-*

# Cross-Topic
## WA
### No Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion False \
    --result_file_name strength_regression_arg_rank_30k_wa-no_motion-cross_topic_ukp_is_arg_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/strength_regression_arg_rank_30k_wa-no_motion-cross_topic/checkpoint-*

### Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task STRENGTH_REGRESSION_ARGUMENT_CORRELATION \
    --include_motion True \
    --result_file_name strength_regression_arg_rank_30k_wa-motion-cross_topic_ukp_is_arg_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/strength_regression_arg_rank_30k_wa-motion-cross_topic/checkpoint-*
