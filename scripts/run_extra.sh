#!/bin/bash

# Ensure enough space
rm -rd models/*

# Argument Detection Model Average Cross Topics
bash scripts/train/argument_detection_ukp/train_average_cross_topic.sh ${1:-42}
bash scripts/results/average_argument_detection_ukp_cross_topic.sh no_motion ${1:-42}
rm -rd models/*

bash scripts/train/argument_detection_ukp/train_average_cross_topic_with_motion.sh ${1:-42}
bash scripts/results/average_argument_detection_ukp_cross_topic.sh motion ${1:-42}
rm -rd models/*
