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
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_loss": 0.5188487768173218,
    "test_f1_micro": 0.7514147130153597,
    "test_f1_macro": 0.7378382471754692,
    "test_f1_weighted": 0.7617598352983316,
    "test_precision_micro": 0.7514147130153598,
    "test_precision_macro": 0.741263567720623,
    "test_precision_weighted": 0.8164123448279282,
    "test_recall_micro": 0.7514147130153598,
    "test_recall_macro": 0.786641077815494,
    "test_recall_weighted": 0.7514147130153598,
    "test_f1_true": 0.7974975304576886,
    "test_f1_false": 0.6781789638932497,
    "test_precision_true": 0.928680981595092,
    "test_precision_false": 0.5538461538461539,
    "test_recall_true": 0.6987882285054818,
    "test_recall_false": 0.8744939271255061,
    "epochs_trained_total": 3.065693430656934,
    "epochs_trained_best": 1.0218978102189782
}
```

