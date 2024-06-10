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
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.5338402986526489,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5593765377998352,
    "test_loss": 1.0932168364524841,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.029325468465685844,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.17124681174755096,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4771520402366685,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5294793925832736,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.721705739692805,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.6974500289810307,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7413215285839118,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7217057396928052,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7084480362854596,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8338025901444659,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7217057396928052,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.781725575238623,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7217057396928052,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7831154512521656,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.6117846067098958,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9532208588957055,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.4636752136752137,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6645282010157711,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8989229494614748,
    "epochs_trained_total": 4.965116279069767,
    "epochs_trained_best": 1.9252491694352158
}
```

