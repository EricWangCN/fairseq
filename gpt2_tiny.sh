mkdir -p /home/v-zilong/fairseq_checkpoints
fairseq-train /home/v-zilong/fairseq_data/wikipedia_en \
    --ddp-backend pytorch_ddp --fp16 --fp16-init-scale 4 \
    --checkpoint-activations \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 50000 --log-format json --log-interval 100 \
    --save-dir /home/v-zilong/fairseq_checkpoints
