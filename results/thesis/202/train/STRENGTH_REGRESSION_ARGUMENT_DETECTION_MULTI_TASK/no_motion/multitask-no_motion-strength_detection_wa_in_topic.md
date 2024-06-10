# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK \
    --task_head_loss_weights 14.8 1 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_wa_in_topic \
    --run_name multitask-no_motion-strength_detection_wa_in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.43805328011512756,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.38249900937080383,
    "test_loss": 0.8205522894859314,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025844529271125793,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16076233983039856,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.48565095892329696,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5378312798611806,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7968291250733999,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7960036245219312,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7961382446118631,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7968291250733999,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8003238006279317,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8000135530493887,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7968291250733999,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7962549013236946,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7968291250733999,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8089804931910195,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7830267558528429,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7704171048019628,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8302304964539007,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.851607903913212,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7409018987341772,
    "epochs_trained_total": 5.477876106194691,
    "epochs_trained_best": 2.4346116027531957
}
```

