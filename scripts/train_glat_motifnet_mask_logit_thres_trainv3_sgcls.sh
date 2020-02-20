#!/usr/bin/env bash
python models/train_rels_motifnet_mask_logit_thres_train_v3_sgcls.py -m sgcls -p 400 \
-tb_log_dir summaries/motifnet_glat_sgcls_mask_logit_thres_train_v3_1 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_sgcls_mask_logit_thres_train_v3_1 \
-adam \
-b 20 \
| tee ./logs/train_motif_glat_sgcls_mask_logit_thres_train_v3_1.txt \

