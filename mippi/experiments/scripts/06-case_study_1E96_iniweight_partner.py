import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
# from keras.metrics import sparse_top_k_categorical_accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import tensorflow.keras.backend as K
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
from PIL import Image
import io
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
import scipy
from scipy.stats import chi2_contingency
from scipy import stats
from mippiNetbuild_att import *
# sys.path.append('../input/mippi0801')
# from transformer import *
np.random.seed(0)

df_s_path = r'../../data/skempi2_window_with_pssm.dataset'
df_s = pd.read_pickle(df_s_path)
df_s = df_s.drop_duplicates('par_seq')
df_s = df_s.iloc[14:16]
df_s['par_seq_len'] = df_s.par_seq.str.len()
df_s.reset_index(drop=True, inplace=True)
print(df_s.shape)
print(df_s.columns)
print(df_s)
#print(df_s.head(10))

aaDict = {'0':0, 'D':1, 'S':2, 'Q':3, 'K':4,
          'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
          'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
          'A':15, 'V':16, 'E':17, 'Y':18, 'M':19, 'C':20}
max_len = 1024
mut0_c = [[aaDict[x] for x in a] for a in df_s['ori_win']]
mut1_c = [[aaDict[x] for x in a] for a in df_s['mut_win']]
par0_c = [[aaDict[x] for x in a] for a in df_s['par_seq']]
# par1_c = [[aaDict[x] for x in a] for a in df_s['par0']]

window_len = 51
mut0_c = keras.preprocessing.sequence.pad_sequences(mut0_c, maxlen=window_len, padding='post')
mut1_c = keras.preprocessing.sequence.pad_sequences(mut1_c, maxlen=window_len, padding='post')
par0_c = keras.preprocessing.sequence.pad_sequences(par0_c, maxlen=max_len, padding='post')
# par1_c = keras.preprocessing.sequence.pad_sequences(par1_c, maxlen=window_len, padding='post')

# label = np.array(df_s['label'])
# # label = to_categorical(label, num_classes=5)
# # before change column: no_effect:0, disrupting:1, decreasing:2, increasing:3, causing:4
# # label = label[:, [1, 2, 0, 3, 4]]
# print(label.shape)

pssm_win_mut0 = df_s['pssm_mut0_win'].values
pssm_win_mut0 = np.stack(pssm_win_mut0, axis=0).astype('float32')
print(pssm_win_mut0.shape)

pssm_win_mut1 = df_s['pssm_mut1_win'].values
pssm_win_mut1 = np.stack(pssm_win_mut1, axis=0).astype('float32')
print(pssm_win_mut1.shape)

pssm_par0 = df_s['pssm_par0'].values
pssm_par0 = [x[:1024, :] for x in pssm_par0] # restrict par protein length to 1024
pssm_par0 = np.stack(pssm_par0, axis=0).astype('float32')
print(pssm_par0.shape)

data = [mut0_c, mut1_c, par0_c, pssm_win_mut0, pssm_win_mut1, pssm_par0]
data_reverse = [mut1_c, mut0_c, par0_c, pssm_win_mut1, pssm_win_mut0, pssm_par0]
data_no = [mut0_c, mut0_c, par0_c, pssm_win_mut0, pssm_win_mut0, pssm_par0]
'''
# output the attention weight of partner
att51_ori = [np.zeros((df_s.shape[0], 4, 1024, 1024), dtype='float16')] * 3
att51_mut = [np.zeros((df_s.shape[0], 4, 1024, 1024), dtype='float16')] * 3
'''

K.clear_session()
model = build_model()
# test_model = keras.Model()
model.summary()
adam = optimizers.Adam(learning_rate=0.0002)
model.compile(adam, loss=categorical_focal_loss(alpha=[.25, .25, .1, .25], gamma=2.), 
              metrics=['acc', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2acc')])

data_seq = [mut0_c, mut1_c, par0_c, pssm_win_mut0, pssm_win_mut1, pssm_par0]
data_mut = [mut1_c, mut1_c, par0_c, pssm_win_mut1, pssm_win_mut1, pssm_par0]

model_path = './via_att0/bestAcc.h50'
model.load_weights(model_path)
att51_model = keras.Model(model.input, [model.output, model.get_layer('token_and_position_embedding_1').output])
att51_ori = att51_model.predict(data_seq)[1][0].sum(axis=-1)[0:185]
print(att51_ori.shape)

np.save('case_study/1E96_iniweight_stage_dataset.npy', att51_ori)
#att51_ori_sum[1].argsort()[-10:][::-1] #eg. array([ 55, 133, 239, 238, 234, 235,  57, 155, 165,  91], dtype=int64) return the index  