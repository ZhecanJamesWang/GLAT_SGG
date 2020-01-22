#!/usr/bin/env bash
python models/train_rels_motifnet_mask.py -m predcls -p 400 \
-tb_log_dir summaries/motifnet_glat_mask_pred \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_mask_pred \
-adam \
-b 1 \
-nepoch 50 \

