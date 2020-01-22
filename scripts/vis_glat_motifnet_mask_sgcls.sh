#!/usr/bin/env bash
python visualization/visualize_motifnet_mask.py -m sgcls -p 400 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-adam \
-b 1 \
-nepoch 50 \


