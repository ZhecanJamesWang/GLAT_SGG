[Caution: This reposity is still under development mode and not cleanly documented yet. We only recommed you to use it as a reference.]

At this repository, we build our Global-Local-Transformer model on top of a selection of base scene grap generator models including KERN, Neural Motif, Stanford, etc to improve scene graph generation by leveraging Visual Commonsense

The corresponding paper was accepted at ECCV 2020
 arXiv preprint arXiv:2006.09623 (2020).
Alireza Zareian*, Zhecan Wang*, Haoxuan You*, Shih-Fu Chang, "Learning Visual Commonsense for Robust Scene Graph Generation", ECCV, 2020. (* co-first authors) [[manuscript](https://arxiv.org/abs/2006.09623)] 

# Reference to Base Scene Graph Generators

## Knowledge-Embedded Routing Network  for Scene Graph Generation
Tianshui Chen*, Weihao Yu*, Riquan Chen, and Liang Lin, “Knowledge-Embedded Routing Network for Scene Graph Generation”, CVPR, 2019. (* co-first authors) [[manuscript](https://arxiv.org/abs/1903.03326)] 

## Neural Motifs: Scene Graph Parsing with Global Context
Zellers R, Yatskar M, Thomson S, Choi Y. "Neural motifs: Scene graph parsing with global context". CVPR, 2018.

## Scene Graph Generation by Iterative Message Passing
Xu D, Zhu Y, Choy CB, Fei-Fei L. "Scene graph generation by iterative message passing".CVPR 2017.

# Evaluation metrics
In validation/test dataset, assume there are <img src="https://latex.codecogs.com/gif.latex?Y" />  images. For each image, a model generates top <img src="https://latex.codecogs.com/gif.latex?X" /> predicted relationship triplets. As for image <img src="https://latex.codecogs.com/gif.latex?I_y" />, there are <img src="https://latex.codecogs.com/gif.latex?G_y" /> ground truth relationship triplets, where <img src="https://latex.codecogs.com/gif.latex?T_{y}^{X}" /> triplets are predicted successfully by the model. We can calculate:

<img src="https://latex.codecogs.com/gif.latex?R@X=\frac{1}{Y}\sum_{y=1}^{Y}\frac{T_y^X}{G_y}." />


For image <img src="https://latex.codecogs.com/gif.latex?I_y" />, in its <img src="https://latex.codecogs.com/gif.latex?G_y" /> ground truth relationship triplets, there are <img src="https://latex.codecogs.com/gif.latex?G_{yk}" /> ground truth triplets with relationship <img src="https://latex.codecogs.com/gif.latex?k" /> (Except <img src="https://latex.codecogs.com/gif.latex?k=1" />, meaning no relationship. The number of relationship classes is <img src="https://latex.codecogs.com/gif.latex?K" />, including no relationship), where <img src="https://latex.codecogs.com/gif.latex?T_{yk}^X" /> triplets are predicted successfully by the model. In <img src="https://latex.codecogs.com/gif.latex?Y" /> images of validation/test dataset, for relationship <img src="https://latex.codecogs.com/gif.latex?k" />, there are <img src="https://latex.codecogs.com/gif.latex?Y_k" /> images which contain at least one ground truth triplet with this relationship. The R@X of relationship <img src="https://latex.codecogs.com/gif.latex?k" /> can be calculated:


<img src="https://latex.codecogs.com/gif.latex?R@X_k=\frac{1}{Y_k}\sum_{y=1,G_{yk}\neq0}^{Y}\frac{T_{yk}^X}{G_{yk}}." />



Then we can calculate:

<img src="https://latex.codecogs.com/gif.latex?mR@X=\frac{1}{K-1}\sum_{k=2}^{K}R@X_k." />


