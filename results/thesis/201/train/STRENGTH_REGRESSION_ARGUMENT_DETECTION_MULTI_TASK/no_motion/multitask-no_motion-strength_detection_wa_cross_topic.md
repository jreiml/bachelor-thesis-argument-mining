# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK \
    --task_head_loss_weights 18.2 1 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_wa_cross_topic \
    --run_name multitask-no_motion-strength_detection_wa_cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5633313655853271,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5472233891487122,
    "test_loss": 1.1105547547340393,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.030952276661992073,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.17593258619308472,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.46860592478596486,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.518598160566557,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7360549717057397,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7178764198542442,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7503546605825531,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7360549717057397,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.724431401080174,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8217571193031935,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7360549717057397,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7817154662313813,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7360549717057397,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7894906511927788,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.6462621885157097,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9390337423312883,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.5098290598290598,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6810344827586207,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.882396449704142,
    "epochs_trained_total": 4.863787375415282,
    "epochs_trained_best": 1.823920265780731
}
```

