# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_in_topic \
    --run_name multitask-no_motion-detection_evidence_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.43710240721702576,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4758599102497101,
    "test_loss": 0.9129623174667358,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7979911014330401,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7976496768219486,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8054462547822001,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8087116527967816,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8021130623999213,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8044756305708325,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7915065722952478,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7434279705573081,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8674645390070922,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8764462809917355,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7277798438081071,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.041628800332546234,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.20403137803077698,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7702698075130731,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7871851159375083,
    "epochs_trained_total": 5.708777686628384,
    "epochs_trained_best": 2.7041578515608133
}
```

