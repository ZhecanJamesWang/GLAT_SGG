#!/usr/bin/env bash




echo "EVALING STANFORD"
python models/eval_stanford_bal.py -m sgcls -b 6 -clip 5 \
    -p 100 -pooling_dim 4096 -ngpu 1 -test -ckpt checkpoints/stanford_bal/vgrel-7.tar -nepoch 50 -cache ./cache/stanford_sgcls_bal | tee ./logs/eval_stanford_sgcls_bal.txt


#echo "EVALING STANFORD_GLAT"
#python models/eval_stanford_motif_glat.py -m sgcls -model_s_m stanford -b 6 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/stanford_1/vgrel-7.tar -nepoch 50 -cache ./cache/stanford_glat  | tee eval_stanford_glat.txt
#
##python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache motifnet_predcls
