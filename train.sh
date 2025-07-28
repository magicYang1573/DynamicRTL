gpus=0
model=default
train_seq_len=30
eval_seq_len=30
exp_id=${model}_model_${eval_seq_len}

# nohup \
python src/train.py \
    --data_dir dataset_npz \
    --batch_size 64 \
    --graph_npz_name graphs.npz \
    --label_npz_name labels.npz \
    --gpus $gpus \
    --model $model \
    --train_seq_len $train_seq_len \
    --eval_seq_len $eval_seq_len \
    --exp_id $exp_id \
    --supervision default \
    # > output_log/${exp_id}.log 2>&1 &