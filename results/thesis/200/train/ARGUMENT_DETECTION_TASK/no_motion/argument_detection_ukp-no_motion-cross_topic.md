# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
    --task ARGUMENT_DETECTION_TASK \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_detection_ukp-no_motion-cross_topic \
    --run_name argument_detection_ukp-no_motion-cross_topic \
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
    "test_loss": 0.5627599358558655,
    "test_f1_micro": 0.7324171382376718,
    "test_f1_macro": 0.7183447188044025,
    "test_f1_weighted": 0.743079604318214,
    "test_precision_micro": 0.7324171382376717,
    "test_precision_macro": 0.7224296707042106,
    "test_precision_weighted": 0.7948760769917053,
    "test_recall_micro": 0.7324171382376717,
    "test_recall_macro": 0.762259336168529,
    "test_recall_weighted": 0.7324171382376717,
    "test_f1_true": 0.7813016187644533,
    "test_f1_false": 0.6553878188443518,
    "test_precision_true": 0.9068251533742331,
    "test_precision_false": 0.538034188034188,
    "test_recall_true": 0.6863029599535694,
    "test_recall_false": 0.8382157123834887,
    "epochs_trained_total": 3.372262773722628,
    "epochs_trained_best": 1.3284671532846715
}
```

