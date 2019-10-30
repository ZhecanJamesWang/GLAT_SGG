#!/usr/bin/env bash
python models/train_rels.py -m predcls -p 100 -clip 5 \
-tb_log_dir summaries/kern_predcls \
-save_dir checkpoints/kern_predcls \
-ckpt checkpoints/vgdet/vg-faster-rcnn.tar \
-val_size 5000 \
-adam \
-b 1 \
-lr 1e-5 \
