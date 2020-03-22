#!/usr/bin/env bash
python models/train_rels_glat_universal_v3.py -m sgdet  -model_s_m motifnet -p 400 \
-tb_log_dir summaries/train_glat_motif_sgdet_v31 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet-sgdet/vgrel-motifnet-sgdet.tar \
-save_dir checkpoints/train_glat_motif_sgdet_v31 \
-adam \
-b 20 \
| tee ./logs/train_glat_motif_sgdet_v31.txt \

