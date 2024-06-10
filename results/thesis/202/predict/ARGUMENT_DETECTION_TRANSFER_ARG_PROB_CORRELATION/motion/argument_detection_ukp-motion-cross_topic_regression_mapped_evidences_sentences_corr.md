# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION \
    --include_motion True \
    --result_file_name argument_detection_ukp-motion-cross_topic_regression_mapped_evidences_sentences_corr \
    --result_prefix 202 \
    --model_name ./models/argument_detection_ukp-motion-cross_topic/checkpoint-336
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.07677259296178818,
            "root_mean_squared_error": 0.27707868814468384,
            "spearman": 0.5304212528618981,
            "pearson": 0.5297346407579964
        },
        "positive_output": {
            "mean_squared_error": 0.07490701228380203,
            "root_mean_squared_error": 0.27369144558906555,
            "spearman": 0.5322976231777578,
            "pearson": 0.5310513251312888
        },
        "negative_output": {
            "mean_squared_error": 0.07936733961105347,
            "root_mean_squared_error": 0.2817220985889435,
            "spearman": 0.5066998449326845,
            "pearson": 0.49155978469597944
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.07551446557044983,
            "root_mean_squared_error": 0.2747989594936371,
            "spearman": 0.5256643086416924,
            "pearson": 0.5256642055336465
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.07783690094947815,
            "root_mean_squared_error": 0.2789926528930664,
            "spearman": 0.5256643187025898,
            "pearson": 0.5245988888439784
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 2.3807590007781982,
            "root_mean_squared_error": 1.5429707765579224,
            "spearman": 0.5322975565534743,
            "pearson": 0.5292664922691946
        },
        "negative_output": {
            "mean_squared_error": 0.6743582487106323,
            "root_mean_squared_error": 0.8211931586265564,
            "spearman": -0.5066999960027658,
            "pearson": -0.485578218528919
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.09124337881803513,
            "root_mean_squared_error": 0.3020651936531067,
            "spearman": 0.5256645231463295,
            "pearson": 0.525244428668978
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.452704519033432,
            "root_mean_squared_error": 0.6728332042694092,
            "spearman": -0.525664572587382,
            "pearson": -0.5252444288532789
        }
    }
}
```

