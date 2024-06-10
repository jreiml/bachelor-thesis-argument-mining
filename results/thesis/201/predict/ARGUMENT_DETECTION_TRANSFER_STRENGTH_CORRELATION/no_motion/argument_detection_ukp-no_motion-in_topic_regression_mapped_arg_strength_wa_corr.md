# Command
```bash
python3 src/models/predict_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv \
    --per_device_prediction_batch_size 64 \
    --task ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION \
    --include_motion False \
    --result_file_name argument_detection_ukp-no_motion-in_topic_regression_mapped_arg_strength_wa_corr \
    --result_prefix 201 \
    --model_name ./models/argument_detection_ukp-no_motion-in_topic/checkpoint-522
```

# Results
```json
{
    "regression_metrics": {
        "all_outputs": {
            "mean_squared_error": 0.03461971506476402,
            "root_mean_squared_error": 0.1860637366771698,
            "spearman": 0.20449445222936477,
            "pearson": 0.21651367059443835
        },
        "positive_output": {
            "mean_squared_error": 0.03529135882854462,
            "root_mean_squared_error": 0.1878599375486374,
            "spearman": 0.17284624241248947,
            "pearson": 0.16784035354806315
        },
        "negative_output": {
            "mean_squared_error": 0.03560329228639603,
            "root_mean_squared_error": 0.18868835270404816,
            "spearman": 0.13931426845591338,
            "pearson": 0.1362981053197586
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.0355854332447052,
            "root_mean_squared_error": 0.1886410117149353,
            "spearman": 0.15690201831421222,
            "pearson": 0.13891187628156154
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.03554731607437134,
            "root_mean_squared_error": 0.18853995203971863,
            "spearman": 0.15690326362646612,
            "pearson": 0.1422165100701303
        }
    },
    "raw_metrics": {
        "positive_output": {
            "mean_squared_error": 1.1796663999557495,
            "root_mean_squared_error": 1.0861245393753052,
            "spearman": 0.17284719861465542,
            "pearson": 0.16738965141900725
        },
        "negative_output": {
            "mean_squared_error": 4.017337799072266,
            "root_mean_squared_error": 2.0043296813964844,
            "spearman": -0.13931351913837195,
            "pearson": -0.13328045827093984
        },
        "softmax_positive_output": {
            "mean_squared_error": 0.1269616335630417,
            "root_mean_squared_error": 0.3563167452812195,
            "spearman": 0.15690198056122037,
            "pearson": 0.14163644674990733
        },
        "softmax_negative_output": {
            "mean_squared_error": 0.3690834939479828,
            "root_mean_squared_error": 0.6075224280357361,
            "spearman": -0.1569022300775544,
            "pearson": -0.14163644691616054
        }
    }
}
```

