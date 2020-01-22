#!/usr/bin/env bash
python models/train_rels_motifnet.py -m sgcls -p 100 \
-tb_log_dir summaries/motifnet_glat \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat \
-adam \
-b 1 \

