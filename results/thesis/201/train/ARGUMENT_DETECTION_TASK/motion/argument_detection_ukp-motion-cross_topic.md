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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_loss": 0.5939829349517822,
    "test_f1_micro": 0.7348423605497171,
    "test_f1_macro": 0.7133796286239611,
    "test_f1_weighted": 0.7520569298786469,
    "test_precision_micro": 0.7348423605497171,
    "test_precision_macro": 0.7221393083739709,
    "test_precision_weighted": 0.8377939625113607,
    "test_recall_micro": 0.7348423605497171,
    "test_recall_macro": 0.792653978303538,
    "test_recall_weighted": 0.7348423605497171,
    "test_f1_true": 0.7918121231355125,
    "test_f1_false": 0.6349471341124096,
    "test_precision_true": 0.9566717791411042,
    "test_precision_false": 0.4876068376068376,
    "test_recall_true": 0.6754195993502978,
    "test_recall_false": 0.9098883572567783,
    "epochs_trained_total": 3.4744525547445253,
    "epochs_trained_best": 1.4306569343065694
}
```

