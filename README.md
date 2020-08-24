# TLOpt:  Transfer Learning based Inverse Deisgn of Optical Materials 

## Source code for Inverse Design of composite Metal Oxide Optical Materials based on Deep Transfer Learning

by Rongzhi Dong and Jianjun Hu
School of Mechanical Engineering, Guizhou University

Department of Computer Science and Engineering, University of South Carolina
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


### Dataset

Download link: https://www.osti.gov/dataexplorer/biblio/dataset/1479489

Dataset citation:  Stein, Helge S., Edwin Soedarmadji, Paul F. Newhouse, Dan Guevarra, and John M. Gregoire. "Synthesis, optical imaging, and absorption 
spectroscopy data for 179072 metal oxides." Scientific data 6, no. 1 (2019): 1-5. [paper](https://www.nature.com/articles/s41597-019-0019-4)
