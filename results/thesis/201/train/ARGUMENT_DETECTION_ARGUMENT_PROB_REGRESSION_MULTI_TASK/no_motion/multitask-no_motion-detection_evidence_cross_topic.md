# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 7.6 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_cross_topic \
    --run_name multitask-no_motion-detection_evidence_cross_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.5922389626502991,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4319441616535187,
    "test_loss": 1.0241831243038177,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7021018593371059,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.6730908629966758,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7258381290701851,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7021018593371059,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.687985029626134,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8291533267358537,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7021018593371059,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.765248518176328,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7021018593371059,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7704764870756774,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.5757052389176741,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9486196319018405,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.42735042735042733,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.6486628211851075,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8818342151675485,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05651785433292389,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.2377348393201828,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.6664737592071102,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.6707079241985455,
    "epochs_trained_total": 5.252312867956266,
    "epochs_trained_best": 2.222132367212266
}
```

