#!/usr/bin/env bash
python models/train_rels_glat_universal.py -m sgcls  -model_s_m motifnet -p 400 \
-tb_log_dir summaries/train_glat_motif_sgcls_v2 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/train_glat_motif_sgcls_v2 \
-adam \
-b 20 \
| tee ./logs/train_glat_motif_sgcls_v2.txt \

