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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4446525573730469,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.38655567169189453,
    "test_loss": 0.8312082290649414,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02610834501683712,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16158077120780945,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.49444010666388183,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5414339816338665,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.80074378547661,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7998203489009399,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8000784853500466,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.80074378547661,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8037825059101655,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8032887774645124,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.80074378547661,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7997425464325496,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.80074378547661,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8134164222873901,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7862242755144897,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7777777777777778,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8297872340425532,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8524779101037264,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7470071827613727,
    "epochs_trained_total": 5.579243765084473,
    "epochs_trained_best": 2.5360198932202147
}
```

