#!/usr/bin/env bash
python models/train_rels_motifnet_aemask.py -m predcls -p 400 \
-tb_log_dir summaries/motifnet_glat_aemask \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_aemask \
-adam \
-b 20 \
-nepoch 50 \
| tee ./logs/train_motif_glat_predcls_aemask.txt \

