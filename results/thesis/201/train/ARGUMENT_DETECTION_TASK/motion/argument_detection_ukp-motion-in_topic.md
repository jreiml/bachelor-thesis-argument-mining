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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_loss": 0.4163048267364502,
    "test_f1_micro": 0.8181640242708945,
    "test_f1_macro": 0.8175598764503162,
    "test_f1_weighted": 0.8175413821292781,
    "test_precision_micro": 0.8181640242708945,
    "test_precision_macro": 0.8226728956454945,
    "test_precision_weighted": 0.82274086858079,
    "test_recall_micro": 0.8181640242708945,
    "test_recall_macro": 0.818267935544675,
    "test_recall_weighted": 0.8181640242708945,
    "test_f1_true": 0.8280584860262816,
    "test_f1_false": 0.8070612668743509,
    "test_precision_true": 0.784086926042762,
    "test_precision_false": 0.8612588652482269,
    "test_recall_true": 0.8772549019607843,
    "test_recall_false": 0.7592809691285658,
    "epochs_trained_total": 3.8397212543554007,
    "epochs_trained_best": 1.818815331010453
}
```

