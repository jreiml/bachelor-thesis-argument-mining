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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5314881801605225,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5472816824913025,
    "test_loss": 1.078769862651825,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.030024295672774315,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1732752025127411,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.45298148263689025,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5119871751899646,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7653597413096201,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7554715662696452,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.772584568700637,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.76535974130962,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7566878375544019,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8124080549666614,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.76535974130962,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7912048005755337,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.76535974130962,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8046441191317516,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7062990134075386,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9167944785276073,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.5965811965811966,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7169415292353823,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8654680719156851,
    "epochs_trained_total": 5.167774086378738,
    "epochs_trained_best": 2.127906976744186
}
```

