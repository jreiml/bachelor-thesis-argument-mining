# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 14.8 1 9.25 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_evidence_wa_in_topic \
    --run_name multitask-motion-strength_detection_evidence_wa_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.3282049894332886,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4344322979450226,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.37703847885131836,
    "test_loss": 1.1396757662296295,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02545228973031044,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.15953773260116577,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.5020514636058988,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5476489021766057,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7981981429052021,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7981449823851083,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8089709134095502,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8193095391595507,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8086342930147928,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7981992562145234,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.7986721343487599,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7977241514616441,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7167893445495969,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.9011524822695035,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.9016754850088183,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7155931010207673,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03548162057995796,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.1883656531572342,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8051047834762749,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8254152945900354,
    "epochs_trained_total": 4.7298728813559325,
    "epochs_trained_best": 1.710805084745763
}
```

