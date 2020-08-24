import random
import numpy
import pymatgen as mg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matminer.featurizers.composition import ElementProperty
import numpy as np
import pandas  as pd
import argparse
import tensorflow as tf
from Model import network
from matminer.utils.conversions import str_to_composition
import re
from bayes_opt import BayesianOptimization,UtilityFunction
from matplotlib import gridspec
from sklearn.metrics import mean_absolute_error

import warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


number = 508
target_index = np.loadtxt(str(number)+"_test_index.csv").astype(np.float32)
feature_number = 132

data = np.loadtxt(str(number)+"_magpie_target.csv", dtype = str, delimiter = ",")
target = data[:,134:].astype(np.float32)

target_spectrum = target[int(target_index)]
all_elements =['Ag', 'Al', 'Ba', 'Bi', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'In', 'La', 'Lu', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Rb', 'Sb', 'Sc', 'Sm', 'Sn', 'Sr', 'Tb', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
element_count = 3

net = network(
        axis = feature_number, 
        lr = 1e-3
    )
saver = tf.train.Saver(max_to_keep = 1)

def magpie_feature(formula):
    data =[formula]
    df = pd.DataFrame(data,columns=["formula"])
    df["composition"] = df["formula"].transform(str_to_composition)
    ep_feat = ElementProperty.from_preset(preset_name = "magpie")
    df = ep_feat.featurize_dataframe(df, col_id = "composition")
    df.drop(labels = ["composition"],axis = 1, inplace = True)
    return df.iloc[0,1:].to_numpy()



def target(x1,x2,a,b,c):
    compositions = []
    formula = ''
    x3 = max(1-x1-x2, 0)
    ratios = [x1,x2,x3]
    elements = [all_elements[round(a)],all_elements[round(b)],all_elements[round(c)]]
    ratio_sum = 1.0
    for i in range(element_count):
        element = elements[i]
        if i < element_count - 1:
            if ratio_sum > ratios[i]:
                ratio_sum -= ratios[i]
            else:
                ratios[i] = ratio_sum
                ratio_sum -= ratios[i]            
        else:
            ratios[i] = ratio_sum
        ratio = round(ratios[i], 2)
        formula += element + str(ratio)
    print('*****'+ formula +'*****')
    compositions.append(formula)
    mp_feature = magpie_feature(formula)
    mp_feature = mp_feature.reshape(1,feature_number)
    pre = test(
                    model = net, 
                    feature = mp_feature
                )

    loss = mean_absolute_error(target_spectrum, pre[0])
    return -loss

def evalresult(x1,x2,a,b,c):
    formula = ''
    x3 = max(1-x1-x2, 0)
    ratios = [x1,x2,x3]
    elements = [all_elements[round(a)],all_elements[round(b)],all_elements[round(c)]]
    ratio_sum = 1.0
    for i in range(element_count):
        element = elements[i]
        if i < element_count - 1:
            if ratio_sum > ratios[i]:
                ratio_sum -= ratios[i]
            else:
                ratios[i] = ratio_sum
                ratio_sum -= ratios[i]             
        else:
            ratios[i] = ratio_sum
        ratio = round(ratios[i], 2)
        formula += element + str(ratio)
    print(formula)
    print(data[int(target_index),0],"***")
    mp_feature = magpie_feature(formula)
    mp_feature = mp_feature.reshape(1,feature_number)
    pre = test(
                    model = net, 
                    feature = mp_feature
                )

    loss = mean_absolute_error(target_spectrum, pre[0])
    print(loss)
    return pre[0]

def test(model, feature):
    test_pre = sess_global.run(
                    model.pre,
                    feed_dict = {
                            model.x:feature
                        }
            )
    return test_pre


sess_global= 0
def main():
    global sess_global
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
    with tf.Session(
                config = tf.ConfigProto(gpu_options = gpu_options)
                    ) as sess:
        sess_global = sess
        print("session assigned..........")
        saver.restore(sess, "check_point/model.ckpt")
        random.seed(256)
        
        optimizer = BayesianOptimization(target, {'x1': (0.01,1.0),'x2': (0.01,1.0),
                                                    'a':(0,41.4),'b':(0,41.4),'c':(0,41.4)}, random_state=500)
         
        optimizer.maximize(init_points=500, n_iter=100, acq='ucb', kappa=0.1)
        print(optimizer.max)
         
        best_spectrum = evalresult((optimizer.max)["params"]['a'],(optimizer.max)["params"]['b'],(optimizer.max)["params"]['c'],
                                   (optimizer.max)["params"]['x1'],(optimizer.max)["params"]['x2'])

        np.savetxt(str(target_index)+'_'+str(number)+"_bo_best_spectrum_elements_24.csv", best_spectrum, delimiter = ",")
        
if __name__ == "__main__":
    main()
