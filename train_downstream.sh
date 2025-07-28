gpus=0
weight= /path/to/your/weight
train_seq_len=30
eval_seq_len=30
exp_id=${model}_model_${eval_seq_len}

downstream_task=power
assertion=None
model=dynamic
downstream_gnn_layer=3

# nohup \
python src/train_downstream.py \
    --data_dir dataset_npz \
    --batch_size 64 \
    --graph_npz_name graphs.npz \
    --label_npz_name labels.npz \
    --gpus $gpus \
    --pretrain_weight $weight \
    --train_seq_len $train_seq_len \
    --eval_seq_len $eval_seq_len   \
    --exp_id $exp_id \
    --downstream_task $downstream_task \
    --assertion $assertion \
    --power_model $model \
    --downstream_num_rounds $downstream_gnn_layer
    # > output_log/${exp_id}.log 2>&1 &