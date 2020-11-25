# TLOpt:  Transfer Learning based Inverse Deisgn of Optical Materials 

## Source code for Inverse Design of composite Metal Oxide Optical Materials based on Deep Transfer Learning

by Rongzhi Dong^1 and Jianjun Hu^2

^1 School of Mechanical Engineering, Guizhou University

^2 Department of Computer Science and Engineering, University of South Carolina
2020-08-24

### Installation

matminer
tensorflow
Bayesianoptimization

### Code description

1. model_training.py is used to train Model 1 for optical absorption spectrum prediction from composition.
2. model_transfer.py is used to transfer learning parameters from model1 to model2 and fine-tune it using Dataset B.
3. GA_inverse_design.py is used to inverse design formulas with specified elements through GA algorithm. 
4. BO_inverse_design.py is used to inverse design formulas with specified elements through Bayesian optimization method.
5. GA_inverse_design_elements.py is used to inverse design formulas without specified elements through GA algorithm.
6. BO_inverse_design_elements.py is used to inverse design formulas without specified elements through Bayesiaon optimization BO method.

### Cite our paper

Dong, Rongzhi, Yabo Dan, Xiang Li, and Jianjun Hu. "Inverse design of composite metal oxide optical materials based on deep transfer learning and global optimization." Computational Materials Science (2020): 110166.

@article{dong2020inverse,
  title={Inverse design of composite metal oxide optical materials based on deep transfer learning and global optimization},
  author={Dong, Rongzhi and Dan, Yabo and Li, Xiang and Hu, Jianjun},
  journal={Computational Materials Science},
  pages={110166},
  year={2020},
  publisher={Elsevier}
}


### Dataset

Download link: https://www.osti.gov/dataexplorer/biblio/dataset/1479489
Stein, Helge S., Edwin Soedarmadji, Paul F. Newhouse, Dan Guevarra, and John M. Gregoire. "Synthesis, optical imaging, and absorption 
spectroscopy data for 179072 metal oxides." Scientific data 6, no. 1 (2019): 1-5. [paper](https://www.nature.com/articles/s41597-019-0019-4)
