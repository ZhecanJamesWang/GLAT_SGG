#!/usr/bin/env bash
python models/train_rels_glat_visfea.py -m sgcls -model_s_m motifnet -p 400 \
-tb_log_dir summaries/motif_glat_visfea_train \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motif_glat_visfea_train \
-adam \
-b 20 \
| tee ./logs/train_motif_glat_sgcls_visfea.txt \

