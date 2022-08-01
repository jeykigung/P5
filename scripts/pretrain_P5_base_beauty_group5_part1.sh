# Run with $ bash scripts/pretrain_P5_base_beauty_group5_part1.sh 4

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=beauty-base-group5-part1

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12021 \
    src/pretrain_group5_part1.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train beauty \
        --valid beauty \
        --batch_size 16 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'traditional' \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log
