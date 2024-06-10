# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 14.8 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_evidence_wa_in_topic \
    --run_name multitask-no_motion-strength_detection_evidence_wa_in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4046156704425812,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.43910571932792664,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.38954880833625793,
    "test_loss": 1.2332701981067657,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.026362255215644836,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16236457228660583,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4950218761066452,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5431434429122559,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8038755137991779,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8025474498501908,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8033113243411356,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8038755137991779,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8052880133640588,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8047178083470802,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8038755137991779,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8017909828826822,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8038755137991779,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8187409551374819,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7863539445628998,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7932001402032948,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8173758865248227,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.845981308411215,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7576006573541495,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.038534823805093765,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.1963028907775879,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7803711575651909,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8030104835637203,
    "epochs_trained_total": 5.635593220338983,
    "epochs_trained_best": 2.616525423728813
}
```

