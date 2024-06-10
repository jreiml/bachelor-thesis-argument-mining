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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.38572958111763,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4595891237258911,
    "test_loss": 0.8453187048435211,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.026047205552458763,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1613914668560028,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4784399418936637,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5318816216147548,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7752984928557449,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7752643199961744,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.77565649424172,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7752984928557448,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.78749226271711,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8022595920467032,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7752984928557448,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7893615863418771,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7752984928557448,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7724930638129212,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7780355761794278,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.6831405538030144,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8918439716312057,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8887368901048791,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.6899862825788752,
    "epochs_trained_total": 4.666130329847144,
    "epochs_trained_best": 1.6230018538598763
}
```

