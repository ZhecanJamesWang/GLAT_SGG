#!/usr/bin/env bash
python models/train_rels_motifnet.py -m predcls -p 100 \
-tb_log_dir summaries/motifnet_glat_predcls_mbz \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_predcls_mbz \
-adam \
-b 6 \

