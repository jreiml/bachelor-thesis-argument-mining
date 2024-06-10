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
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5362170934677124,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.434951514005661,
    "test_loss": 0.9711686074733734,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.029469240456819534,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.17166607081890106,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.46761152975370973,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5215943219369277,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8080032336297495,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8042404273710579,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8102960222830046,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8080032336297494,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8028492213308165,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8240806749801523,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8080032336297494,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8177807596047574,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8080032336297494,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8313809016684416,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7770999530736744,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.8980061349693251,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.7076923076923077,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7739590218109715,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8616024973985432,
    "epochs_trained_total": 4.965116279069767,
    "epochs_trained_best": 1.9252491694352158
}
```

