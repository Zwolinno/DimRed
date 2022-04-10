import parser
import argparse
import numpy as np
import random
import sys
import json
import glob, os
import scipy.misc as misc
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.linear_model import SGDClassifier, SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt

# def ParseArguments():
#     parser = argparse.ArgumentParser(description="Project ")
#     parser.add_argument('--dataset', default="", required=False, help='mnist, wine , iris (default: %(default)s)')
#     parser.add_argument('--data-dir', default="", required=False, help='data dir (images)  (default: %(default)s)')
#     parser.add_argument('--data2-dir', default="", required=False, help='data dir (text files) (default: %(default)s)')
#     parser.add_argument('--option', default="bw", required=False, help='color or bw  (default: %(default)s)')
#     parser.add_argument('--normalize', default="no", required=False, help='each column: mean 0, std 1 (default: %(default)s)')
#     parser.add_argument('--test_size', default="0.2", required=False, help='test set size (0,1) (random split) (default: %(default)s)')
#     parser.add_argument('--dim', default="0", required=False, help='PCA reduction (0 = no pca) (default: %(default)s)')
#     parser.add_argument('--algs', default="nb", required=False, help='nb (naive bayes), all = all (default: %(default)s)')
#     parser.add_argument('--knn', default="5", required=False, help='k for knn clf (default: %(default)s)')
#
#     args = parser.parse_args()
#
#     return args.dataset, args.data_dir, args.data2_dir, args.option,  args.normalize, args.test_size, args.dim, args.algs, args.knn

df = pd.read_csv('ratings.csv')
# df['split'] = np.random.randn(df.shape[0], 1)
# msk = np.random.rand(len(df)) <= 0.7
# train = df[msk]
# test = df[~msk]

rng = np.random.RandomState()
train = df.sample(frac=0.9, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

#print(df[['userId', 'movieId', 'rating']])
Z = np.array(train[['userId', 'movieId', 'rating']])

def r_model(Z, r):
    model = NMF(n_components=r, init='random', random_state=0)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approximated = np.dot(W, H)
    return mean_squared_error(Z, Z_approximated, squared=False)

def r_model(train, test, r):
    Z = np.array(train[['userId', 'movieId', 'rating']])
    model = NMF(n_components=r, init='random', random_state=0)
    W = model.fit_transform(train)
    H = model.components_
    Z_approximated = np.dot(W, H)
    #Z_test = Z_approximated
    return mean_squared_error(Z, Z_approximated, squared=False)

rs = [j for j in range(2, 11)]
rmse = [r_model(Z, i) for i in rs]

plt.plot(rs, rmse)
plt.show()
