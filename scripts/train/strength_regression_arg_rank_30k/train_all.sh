#!/bin/bash

# In-Topic
## WA
bash scripts/train/strength_regression_arg_rank_30k/weighted_average.sh False data/processed/arg_quality_rank_30k_wa.csv -no_motion-in_topic ${1:-42}
bash scripts/train/strength_regression_arg_rank_30k/weighted_average.sh True data/processed/arg_quality_rank_30k_wa.csv -motion-in_topic ${1:-42}

# Cross-Topic
## WA
bash scripts/train/strength_regression_arg_rank_30k/weighted_average.sh False data/processed/arg_quality_rank_30k_cross_topic_wa.csv -no_motion-cross_topic ${1:-42}
bash scripts/train/strength_regression_arg_rank_30k/weighted_average.sh True data/processed/arg_quality_rank_30k_cross_topic_wa.csv -motion-cross_topic ${1:-42}
