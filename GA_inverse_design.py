import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pymatgen as mg
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import pandas  as pd
import argparse
import tensorflow as tf
from Model import network
from matminer.utils.conversions import str_to_composition
import re
from sklearn.metrics import mean_absolute_error

number = 508

import warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def binary2dec(bstring):
    if bstring =='':
        return 0
    dec =int(bstring,2) 
    return(dec)


data = np.loadtxt(str(number)+"_target.csv", dtype = str, delimiter = ",")
target_index = np.loadtxt(str(number)+"_magpie_test_index.csv")
target = data[:,134:].astype(np.float32)
target_spectrum = target[int(target_index)]
np.savetxt(str(target_index)+ "_"+ str(number)+"_target_spectrum.csv", target_spectrum, delimiter = ",")
elements =  np.array(re.findall(r"\D+\D*",data[0,0]))
elements = elements[elements != '.']   
ratios = np.arange(0.01,1.0,0.01)

gene_length = 7 
element_count = len(elements) 
feature_number =132

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n = gene_length*element_count)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

net = network(
        axis = 132, 
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

def evalresult(individual):
    formula = ''
    ratio_sum = 1
    for i in range(element_count):
        gene = individual[i*gene_length:(i+1)*gene_length]
        gene=''.join([str(c) for c in gene])
        ratio_i=binary2dec(gene[0:])
        element = elements[i]
        if i < element_count - 1:
            ratio = ratios[ratio_i%len(ratios)]
            if ratio_sum > ratio:
                ratio_sum -= ratio
            else:
                ratio = ratio_sum
                ratio_sum -= ratio
        else:
            ratio = ratio_sum
        ratio = round(ratio, 2)
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
    return pre[0], formula,

def evalOneMax(individual):
    formula = ''
    ratio_sum = 1
    for i in range(len(elements)):
        gene = individual[i*gene_length:(i+1)*gene_length]
        gene=''.join([str(c) for c in gene])
        ratio_i=binary2dec(gene[0:])
        element = elements[i]        
        if i < element_count - 1:
            ratio = ratios[ratio_i%len(ratios)]
            if ratio_sum > ratio:
                ratio_sum -= ratio
            else:
                ratio = ratio_sum
                ratio_sum -= ratio
        else:
            ratio = ratio_sum
        ratio = round(ratio, 2)
        formula += element + str(ratio)
    mp_feature = magpie_feature(formula)
    mp_feature = mp_feature.reshape(1,132)
    pre = test(
                    model = net, 
                    feature = mp_feature
                )
    loss = mean_absolute_error(target_spectrum, pre[0])
    return -loss,


def cxTwoPointCopy(ind1, ind2):   
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: 
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()       
    return ind1, ind2
    
    
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit)
toolbox.register("select", tools.selTournament, tournsize = 3)

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
        
        pop = toolbox.population(n = 500)
        hof = tools.HallOfFame(100, similar = numpy.array_equal)       
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.5, ngen = 100, stats = stats,
                            halloffame = hof)

        best_spectrum = evalresult(hof[0])[0]
        np.savetxt(str(target_index)+"_"+ str(number)+"_ga_best_spectrum.csv", best_spectrum, delimiter = ",")
        return pop,stats,hof

if __name__ == "__main__":
    main()
