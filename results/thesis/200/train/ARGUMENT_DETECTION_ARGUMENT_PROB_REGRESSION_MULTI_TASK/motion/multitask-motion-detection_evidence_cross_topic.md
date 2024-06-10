# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 10 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-detection_evidence_cross_topic \
    --run_name multitask-motion-detection_evidence_cross_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.37378787994384766,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4848589301109314,
    "test_loss": 0.858646810054779,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7738480194017785,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7625567345497857,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7823347960606907,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7738480194017785,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.7642788789261181,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8317627427283495,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7738480194017785,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8085171523500707,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7738480194017785,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8143354902936785,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.710777978805893,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9409509202453987,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.5876068376068376,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7177537291605732,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.8992805755395683,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.037378787994384766,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19333595037460327,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7900389023408882,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7986292907520478,
    "epochs_trained_total": 5.95878889823381,
    "epochs_trained_best": 2.9288962381149237
}
```

