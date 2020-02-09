#previous lr
##!/usr/bin/env bash
#python models/train_rels_motifnet_mask_top_gt.py -m predcls -p 100 \
#-tb_log_dir summaries/motifnet_glat_predcls_mask_top_gt \
#-lr 1e-4 \
#-ngpu 1 \
#-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
#-save_dir checkpoints/motifnet_glat_predcls_mask_top_gt \
#-adam \
#-b 20 \
#| tee ./logs/train_motif_glat_predcls_mask_top_gt.txt \


# glat lr
#!/usr/bin/env bash
python models/train_rels_motifnet_mask_top_gt.py -m predcls -p 100 \
-tb_log_dir summaries/motifnet_glat_predcls_mask_top_gt_glatlr \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar \
-save_dir checkpoints/motifnet_glat_predcls_mask_top_gt_glatlr \
-adam \
-b 20 \
-nepoch 150 \
| tee ./logs/train_motif_glat_predcls_mask_top_gt_glatlr.txt \
