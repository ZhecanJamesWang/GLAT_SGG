#!/usr/bin/env bash
python models/train_rels_motifnet_mask_logit_thres_mbz.py -m predcls -p 100 \
-tb_log_dir summaries/motifnet_glat_predcls_mask_logit_thres_mbz \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_predcls_mask_logit_thres_mbz \
-adam \
-b 20 \
| tee ./logs/train_motif_glat_predcls_mask_logit_thres_mbz.txt \

