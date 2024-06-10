# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.08255958557128906,
            "root_mean_squared_error": 0.28733184933662415,
            "spearman": 0.48737870559057545,
            "pearson": 0.4749222004173865
        },
        "positive_output": {
            "mean_squared_error": 0.08276232331991196,
            "root_mean_squared_error": 0.2876844108104706,
            "spearman": 0.4848070472843563,
            "pearson": 0.47274680417315823
        },
        "negative_output": {
            "mean_squared_error": 0.08533593267202377,
            "root_mean_squared_error": 0.29212313890457153,
            "spearman": 0.4429889539226322,
            "pearson": 0.4467986483098611
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.08608732372522354,
            "root_mean_squared_error": 0.2934064269065857,
            "spearman": 0.47120850841768563,
            "pearson": 0.4393292432884066
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.08645652234554291,
            "root_mean_squared_error": 0.2940348982810974,
            "spearman": 0.47120841945485314,
            "pearson": 0.43534829814203735
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.7819337844848633,
            "root_mean_squared_error": 1.6679129600524902,
            "spearman": 0.4848070803785866,
            "pearson": 0.47403523537823017
        },
        "negative_output": {
            "mean_squared_error": 1.323645830154419,
            "root_mean_squared_error": 1.1504980325698853,
            "spearman": -0.4429889539226322,
            "pearson": -0.44534469391255554
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.10570182651281357,
            "root_mean_squared_error": 0.32511818408966064,
            "spearman": 0.47120842673679586,
            "pearson": 0.4375094115568157
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.49205613136291504,
            "root_mean_squared_error": 0.7014671564102173,
            "spearman": -0.4712083689257645,
            "pearson": -0.437509409484404
        }
    }
}
```

