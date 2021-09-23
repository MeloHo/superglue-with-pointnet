# superglue-with-pointnet
This is a team project for course: 16824-Visual Learning and Recognition, Sping 2021 at CMU

Team member: Yidong He, Bassam Bikdash.

## Main idea
Use PointNet++ with SuperGlue for point cloud registration.
* PointNet++ is a nice feature extractor for point cloud data. The extracted feature points are good candidates for feature matching.
* SuperGlue is a GNN-based feature matching middle-end. It is previously used for matching 2D image point, but in this work it could be used for matching 3D points without much effort.
* The matched points are normalized with Sinkhorn algo and the problem is well defined as an Optimal Transport problem.

## Post and Video
- Detailed explanation could be found [here](https://sites.google.com/d/1sfaHDhkbMFy1p9WOEfm5Gj0vHxXl_E_O/p/1Kl4UNosTZDfdKyvNB4ooy1R95liR0D8r/edit)
- A 3:30 min video could be found [here](https://www.youtube.com/watch?v=i_QSMtEJ_tQ)

## References
[1]: Pan, Yue, et al. "Iterative global similarity points: A robust coarse-to-fine integration solution for pairwise 3d point cloud registration." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.

[2]: Yew, Zi Jian, and Gim Hee Lee. "Rpm-net: Robust point matching using learned features." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[3]: Qi, Charles R., et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." arXiv preprint arXiv:1706.02413 (2017).

[4]: Sarlin, Paul-Edouard, et al. "Superglue: Learning feature matching with graph neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[5]: [Orthogonal_Procrustes_Problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
