#!/bin/bash

# In-Topic
## No Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-*

## Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-*


# Cross-Topic
## No Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/argument_detection_ukp-no_motion-cross_topic/checkpoint-*

## Motion
python3 src/models/predict_model.py --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_alpha_mapped_evidences_sentences_corr \
    --result_prefix ${1:-42} \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-*
