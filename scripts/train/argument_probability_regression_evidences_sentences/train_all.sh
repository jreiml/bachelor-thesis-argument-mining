#!/bin/bash

# In-Topic
## No Motion
bash scripts/train/argument_probability_regression_evidences_sentences/train.sh False data/processed/evidences_sentences.csv -no_motion-in_topic ${1:-42}
## Motion
bash scripts/train/argument_probability_regression_evidences_sentences/train.sh True data/processed/evidences_sentences.csv -motion-in_topic ${1:-42}

# Cross-Topic
## No Motion
bash scripts/train/argument_probability_regression_evidences_sentences/train.sh False data/processed/evidences_sentences_cross_topic.csv -no_motion-cross_topic ${1:-42}
## Motion
bash scripts/train/argument_probability_regression_evidences_sentences/train.sh True data/processed/evidences_sentences_cross_topic.csv -motion-cross_topic ${1:-42}
