# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --task ARGUMENT_DETECTION_TASK \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_detection_ukp-motion-cross_topic \
    --run_name argument_detection_ukp-motion-cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 20 \
    --max_len_percentile 99 \
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_loss": 0.734043300151825,
    "test_f1_micro": 0.6707760711398545,
    "test_f1_macro": 0.6212657319975117,
    "test_f1_weighted": 0.7128695456147026,
    "test_precision_micro": 0.6707760711398545,
    "test_precision_macro": 0.6531087777253421,
    "test_precision_weighted": 0.8713130359418937,
    "test_recall_micro": 0.6707760711398545,
    "test_recall_macro": 0.7763088207505742,
    "test_recall_weighted": 0.6707760711398545,
    "test_f1_true": 0.758200979664539,
    "test_f1_false": 0.48433048433048437,
    "test_precision_true": 0.9792944785276073,
    "test_precision_false": 0.3269230769230769,
    "test_recall_true": 0.6185517074352144,
    "test_recall_false": 0.9340659340659341,
    "epochs_trained_total": 3.781021897810219,
    "epochs_trained_best": 1.7372262773722629
}
```

