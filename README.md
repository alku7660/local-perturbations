# local-perturbations

Instructions:\
(1) Open local_perturbations_example.py to run a training of local explainability models with Feat. Van. R., Feat. Van. R. FI, LID Van. R., Opt. LID Van. R., Feat. LID Van. R., Feat. LID Van. R. as local perturbation methods\
(2) local_perturbations.py contain all the local perturbation algorithms proposed (to check LEAP algorithm, refer to: Jia, Y., Bailey, J., Ramamohanarao, K., Leckie, C., Houle, M.E.: Improving thequality of explanations with local embedding perturbations. In: Proceedings of the25th ACM SIGKDD International Conference on Knowledge Discovery & DataMining. p. 875–884. KDD ’19, Association for Computing Machinery, New York,NY, USA (2019))\
(3) support_local_perturbations.py contains main imports and other various functions used by the local perturbation algorithms\

Synth4 is the synthetic dataset used for the example (other datasets require adjustments on the initial parameters of index_feat_values, and the functions perturbator_function, single_feat_perturbation and feature_verification (line 166, line 193 and line 246 in support_local_perturbations.py).
