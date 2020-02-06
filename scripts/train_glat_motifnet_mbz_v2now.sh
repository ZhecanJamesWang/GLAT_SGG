#!/usr/bin/env bash
python models/train_rels_motifnet_save_base.py -m predcls -p 100 \
-tb_log_dir summaries/motifnet_glat_predcls_mbz_v2 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_predcls_mbz_v2 \
-adam \
-b 20 | tee ./logs/train_rels_stanford_2020_0125_1.txt
