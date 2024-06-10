# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-in_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-in_topic/checkpoint-580
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.07740142941474915,
            "root_mean_squared_error": 0.2782111167907715,
            "spearman": 0.5417805824303499,
            "pearson": 0.5232114359734218
        },
        "positive_output": {
            "mean_squared_error": 0.0772193968296051,
            "root_mean_squared_error": 0.27788376808166504,
            "spearman": 0.5420143160342689,
            "pearson": 0.5250457521542689
        },
        "negative_output": {
            "mean_squared_error": 0.0779995322227478,
            "root_mean_squared_error": 0.27928397059440613,
            "spearman": 0.5327496671536126,
            "pearson": 0.5177894175189982
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.07833318412303925,
            "root_mean_squared_error": 0.27988067269325256,
            "spearman": 0.5396402019854739,
            "pearson": 0.5149709437154353
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.07837305217981339,
            "root_mean_squared_error": 0.27995187044143677,
            "spearman": 0.5396402939156832,
            "pearson": 0.5169792344073174
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.6789400577545166,
            "root_mean_squared_error": 1.2957391738891602,
            "spearman": 0.5420141894064691,
            "pearson": 0.520762668229018
        },
        "negative_output": {
            "mean_squared_error": 0.945262610912323,
            "root_mean_squared_error": 0.9722461700439453,
            "spearman": -0.5327497489813762,
            "pearson": -0.4838863141769181
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.13170260190963745,
            "root_mean_squared_error": 0.3629085421562195,
            "spearman": 0.5396402938996668,
            "pearson": 0.5163602340457198
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.38595297932624817,
            "root_mean_squared_error": 0.6212511658668518,
            "spearman": -0.5396402761866789,
            "pearson": -0.5163602342640636
        }
    }
}
```

