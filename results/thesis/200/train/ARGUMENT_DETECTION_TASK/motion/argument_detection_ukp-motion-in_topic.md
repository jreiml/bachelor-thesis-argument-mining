# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
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
    --output_dir models/argument_detection_ukp-motion-in_topic \
    --run_name argument_detection_ukp-motion-in_topic \
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
    "test_loss": 0.4289510250091553,
    "test_f1_micro": 0.8046584458798199,
    "test_f1_macro": 0.8045219581181411,
    "test_f1_weighted": 0.8041913544287413,
    "test_precision_micro": 0.8046584458798199,
    "test_precision_macro": 0.8126673303950303,
    "test_precision_weighted": 0.8170541063355224,
    "test_recall_micro": 0.8046584458798199,
    "test_recall_macro": 0.8096665844968669,
    "test_recall_weighted": 0.8046584458798199,
    "test_f1_true": 0.8096872616323417,
    "test_f1_false": 0.7993566546039406,
    "test_precision_true": 0.7441289870311952,
    "test_precision_false": 0.8812056737588653,
    "test_recall_true": 0.8879130071099958,
    "test_recall_false": 0.731420161883738,
    "epochs_trained_total": 3.940766550522648,
    "epochs_trained_best": 1.9198606271777003
}
```

