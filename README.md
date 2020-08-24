
## Source code for Inverse Design of composite Metal Oxide Optical Materials based on Deep Transfer Learning

by Rongzhi Dong and Jianjun Hu\\
2020-08-24



1. model_training.py is used to train Model 1 for optical absorption spectrum prediction from composition.
2. model_transfer.py is used to transfer learning parameters from model1 to model2 and fine-tune it using Dataset B.
3. GA_inverse_design.py is used to inverse design formulas with specified elements through GA algorithm. 
4. BO_inverse_design.py is used to inverse design formulas with specified elements through Bayesian optimization method.
5. GA_inverse_design_elements.py is used to inverse design formulas without specified elements through GA algorithm.
6. BO_inverse_design_elements.py is used to inverse design formulas without specified elements through Bayesiaon optimization BO method.


