#!/bin/bash

# Singletask
bash scripts/results/average.sh train ARGUMENT_DETECTION_TASK ${1:-42}
bash scripts/results/average.sh train ARGUMENT_PROB_REGRESSION_TASK ${1:-42}
bash scripts/results/average.sh train STRENGTH_REGRESSION_TASK ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_DETECTION_STRENGTH_CORRELATION ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION ${1:-42}
bash scripts/results/average.sh predict ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION ${1:-42}
bash scripts/results/average.sh predict STRENGTH_REGRESSION_ARGUMENT_CORRELATION ${1:-42}
bash scripts/results/average.sh predict STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION ${1:-42}

# Multitask
bash scripts/results/average.sh train ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK ${1:-42}
bash scripts/results/average.sh train STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK ${1:-42}
bash scripts/results/average.sh train STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK ${1:-42}
bash scripts/results/average.sh train ALL_MULTI_TASK ${1:-42}