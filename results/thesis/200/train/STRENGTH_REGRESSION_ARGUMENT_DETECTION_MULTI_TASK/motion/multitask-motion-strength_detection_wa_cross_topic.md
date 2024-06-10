# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK \
    --task_head_loss_weights 18.2 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_wa_cross_topic \
    --run_name multitask-motion-strength_detection_wa_cross_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4144161641597748,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5142968893051147,
    "test_loss": 0.9287130534648895,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02823876030743122,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16804392635822296,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4817589112746426,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5363680708998336,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8164915117219078,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8128306137684638,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8187346052656368,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8164915117219078,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8112966021708353,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8329291359730628,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8164915117219078,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8270191031227105,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8164915117219078,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8390070921985817,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7866541353383459,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9072085889570553,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.7153846153846154,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7803430079155673,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8736951983298539,
    "epochs_trained_total": 4.965116279069767,
    "epochs_trained_best": 1.9252491694352158
}
```

