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
import numpy as np
from PIL import Image
import io

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

df_path = r'../../data/processed_mutations.dataset'
# df_path = r'../../data/skempi2_window_with_pssm.dataset'
df = pd.read_pickle(df_path)
pd.set_option('display.max_columns', None)
print(df)

aaDict = {'0':0, 'D':1, 'S':2, 'Q':3, 'K':4,
          'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
          'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
          'A':15, 'V':16, 'E':17, 'Y':18, 'M':19, 'C':20}
max_len = 1024
mut0_c = [[aaDict[x] for x in a] for a in df['mut0_51']]
print(mut0_c)
mut1_c = [[aaDict[x] for x in a] for a in df['mut1_51']]
par0_c = [[aaDict[x] for x in a] for a in df['par0']]
# par1_c = [[aaDict[x] for x in a] for a in df['par0']]

window_len = 51
mut0_c = keras.preprocessing.sequence.pad_sequences(mut0_c, maxlen=window_len, padding='post')
mut1_c = keras.preprocessing.sequence.pad_sequences(mut1_c, maxlen=window_len, padding='post')
par0_c = keras.preprocessing.sequence.pad_sequences(par0_c, maxlen=max_len, padding='post')
# par1_c = keras.preprocessing.sequence.pad_sequences(par1_c, maxlen=window_len, padding='post')

# label = np.array(df['label'])
# # label = to_categorical(label, num_classes=5)
# # before change column: no_effect:0, disrupting:1, decreasing:2, increasing:3, causing:4
# # label = label[:, [1, 2, 0, 3, 4]]
# print(label.shape)

pssm_win_mut0 = df['pssm_win_mut0'].values
print(pssm_win_mut0.shape) #(16505,)
pssm_win_mut0 = np.stack(pssm_win_mut0, axis=0).astype('float32')
print(pssm_win_mut0.shape) #(16505, 51, 20)

pssm_win_mut1 = df['pssm_win_mut1'].values
pssm_win_mut1 = np.stack(pssm_win_mut1, axis=0).astype('float32')
print(pssm_win_mut1.shape) #(16505, 51, 20)

pssm_par0 = df['pssm_par0'].values
pssm_par0 = [x[:1024, :] for x in pssm_par0] # restrict par protein length to 1024
pssm_par0 = np.stack(pssm_par0, axis=0).astype('float32')
print(pssm_par0.shape) #(16505, 1024, 20)

data = [mut0_c, mut1_c, par0_c, pssm_win_mut0, pssm_win_mut1, pssm_par0]
#print(data)
# data = [mut0_c[:100], mut1_c[:100], par0_c[:100], pssm_win_mut0[:100], pssm_win_mut1[:100], pssm_par0[:100]]
data_reverse = [mut1_c, mut0_c, par0_c, pssm_win_mut1, pssm_win_mut0, pssm_par0]
data_no = [mut0_c, mut0_c, par0_c, pssm_win_mut0, pssm_win_mut0, pssm_par0]

K.clear_session()
model = build_model()
# test_model = keras.Model()
model.summary()
adam = optimizers.Adam(learning_rate=0.0002)

# model_path = r'./pssm_in_trans/bestAcc.h53'
# model_path = r'/lustre/home/acct-bmelgn/bmelgn-2/QianWei/MIPPI2/src/kaggle/cross_validation/activation_test/s51_leaky_3block_wfl_gp_HE/bestAcc.h51'

# model_path = r'bestAcc.h54'
# model.load_weights(model_path)
model.compile(adam, loss=categorical_focal_loss(alpha=[.25, .25, .1, .25], gamma=2.), 
              metrics=['acc', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2acc')])

print(data[0].shape[0]) #16505

layer_name = ['_1', '_3', '_5']
seq1 = 16500
step = 5
seq2 = seq1 + step
while(seq2 <= data[0].shape[0]):
    att51_ori = [np.zeros((step, 4, 1024, 1024), dtype='float16')] * 3
    att51_mut = [np.zeros((step, 4, 1024, 1024), dtype='float16')] * 3
    data_ori = [mut0_c[seq1:seq2], mut0_c[seq1:seq2], par0_c[seq1:seq2], pssm_win_mut0[seq1:seq2], pssm_win_mut0[seq1:seq2], pssm_par0[seq1:seq2]]
    data_mut = [mut1_c[seq1:seq2], mut1_c[seq1:seq2], par0_c[seq1:seq2], pssm_win_mut1[seq1:seq2], pssm_win_mut1[seq1:seq2], pssm_par0[seq1:seq2]]
    for i in range(3):
        for j in range(5):
            model_path = './via_att0/bestAcc.h5' + str(j)
            model.load_weights(model_path)
            att51_model = keras.Model(model.input, [model.output, model.get_layer('token_and_position_embedding_1').output])
            att51_ori[i] = att51_model.predict(data_ori)[1][1] + att51_ori[i]
            att51_mut[i] = att51_model.predict(data_mut)[1][1] + att51_mut[i]
            

    partner_wei_sum = att51_ori[0].sum(axis=-2)
    print(partner_wei_sum.shape) #(step, 4, 1024)
    for t in range(step):
        print(partner_wei_sum[t].shape)
        np.save('att_weight/ini_weight_partner/partner_' + str(seq1 + t) + '.npy', partner_wei_sum[t])        
        
        
    seq1 += step
    seq2 += step