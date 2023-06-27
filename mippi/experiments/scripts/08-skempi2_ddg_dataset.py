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
from mippiNetbuild_att import *
# sys.path.append('../input/mippi0801')
# from transformer import *
np.random.seed(0)

# df_path = r'../../../data/raw/raw_s51_0805_p.csv'
df_path = r'../../data/skempi2_window_with_pssm.dataset'
df = pd.read_pickle(df_path)
print(df.shape)

aaDict = {'0':0, 'D':1, 'S':2, 'Q':3, 'K':4,
          'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
          'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
          'A':15, 'V':16, 'E':17, 'Y':18, 'M':19, 'C':20}
max_len = 1024
mut0_c = [[aaDict[x] for x in a] for a in df['ori_win']]
mut1_c = [[aaDict[x] for x in a] for a in df['mut_win']]
par0_c = [[aaDict[x] for x in a] for a in df['par_seq']]
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

pssm_win_mut0 = df['pssm_mut0_win'].values
pssm_win_mut0 = np.stack(pssm_win_mut0, axis=0).astype('float32')
print(pssm_win_mut0.shape)

pssm_win_mut1 = df['pssm_mut1_win'].values
pssm_win_mut1 = np.stack(pssm_win_mut1, axis=0).astype('float32')
print(pssm_win_mut1.shape)

pssm_par0 = df['pssm_par0'].values
pssm_par0 = [x[:1024, :] for x in pssm_par0] # restrict par protein length to 1024
pssm_par0 = np.stack(pssm_par0, axis=0).astype('float32')
print(pssm_par0.shape)

data = [mut0_c, mut1_c, par0_c, pssm_win_mut0, pssm_win_mut1, pssm_par0]
data_reverse = [mut1_c, mut0_c, par0_c, pssm_win_mut1, pssm_win_mut0, pssm_par0]
data_no = [mut0_c, mut0_c, par0_c, pssm_win_mut0, pssm_win_mut0, pssm_par0]

K.clear_session()
model = build_model()
model.summary()
adam = optimizers.Adam(learning_rate=0.0002)

# model_path = r'./pssm_in_trans/bestAcc.h53'
# model_path = r'/lustre/home/acct-bmelgn/bmelgn-2/QianWei/MIPPI2/src/kaggle/cross_validation/activation_test/s51_leaky_3block_wfl_gp_HE/bestAcc.h51'


# model_path = r'bestAcc.h54'
# model.load_weights(model_path)
model.compile(adam, loss=categorical_focal_loss(alpha=[.25, .25, .1, .25], gamma=2.), 
              metrics=['acc', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2acc')])

score_arr = np.zeros((data[0].shape[0], 4))
score_reverse_arr = np.zeros((data[0].shape[0], 4))
class_arr = np.zeros((data[0].shape[0], 5))
class_reverse_arr = np.zeros((data[0].shape[0], 5))
for i in range(5):
    model_path = r'./via_att0/bestAcc.h5' + str(i)
    model.load_weights(model_path)
    pred = model.predict(data)
    pred_reverse = model.predict(data_reverse)
    pred_no = model.predict(data_no)
    pred_class = pred.argmax(axis=-1)
    pred_reverse_class = pred_reverse.argmax(axis=-1)
    pred_no_class = pred_no.argmax(axis=-1)
    df['pred_class' + str(i)] = pred_class
    df['pred_reverse_class' + str(i)] = pred_reverse_class
    df['pred_no_class' + str(i)] = pred_no_class
    score_arr += pred
    score_reverse_arr += pred_reverse
    
    class_arr[:, i] = pred_class
    class_reverse_arr[:, i] = pred_reverse_class
    
from collections import Counter
consistent_score = np.zeros(df.shape[0])
most_common = np.zeros(df.shape[0])
for i in range(class_arr.shape[0]):
    consistent_score[i] = Counter(class_arr[i]).most_common()[0][1]
    most_common[i] = Counter(class_arr[i]).most_common()[0][0]
df['con_score'] = consistent_score
df['most_common'] = most_common

score_cv5_class = score_arr.argmax(axis=-1)
score_cv5_reverse_class = score_reverse_arr.argmax(axis=-1)
df['cv5_class'] = score_cv5_class
df['cv5_reverse_class'] = score_cv5_reverse_class
df['cv5_score'] = (score_arr / 5).max(axis=-1)
df['cv5_reverse_score'] = (score_reverse_arr / 5).max(axis=-1)
print(df.head())

all_cv5_class = np.c_[df['pred_class0'].values, df['pred_class1'].values, df['pred_class2'].values, df['pred_class3'].values, df['pred_class4'].values]
print(df.groupby('cv5_reverse_class').mean()['ddg'])
print(df.groupby('cv5_class').mean()['ddg'])
df.to_pickle('skempi2_via_att_plots_new/ddg_dataset.pickle')

import matplotlib.pyplot as plt
import seaborn as sns
'''
labels = ['disrupting', 'decreasing','no effect', 'increasing']
plt.figure(figsize=(10, 8), dpi=300)
cm = confusion_matrix(df['cv5_class'], df['cv5_reverse_class'], 
                      normalize='true')
ax = sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                linewidths=0.1, linecolor='silver', annot=True, fmt='.2f', annot_kws={"size": 20}, cbar_kws={ "label": 'proportion'}
                ,cmap='mako')
# ax.set(xlabel='predict label', ylabel='true label')
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=20)
ax.figure.axes[-1].yaxis.label.set_size(20)
# ax.figure.axes[-1].yaxis.label.set_weight('bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15, va='center')
plt.xlabel('reverse label', fontsize=20)
plt.ylabel('original label', fontsize=20)
# plt.savefig('class_transmit.jpg', dpi=300)

png1 = io.BytesIO()
plt.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("./skempi2_via_att_plots_new/class_transmit_tif.tiff", dpi=png2.info['dpi'])
png1.close()
'''

