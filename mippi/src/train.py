#!/usr/bin/env python
# coding: utf-8


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
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
import sys
from sklearn.model_selection import train_test_split
from mippiNetbuild import *
np.random.seed(0)


# id: window? activation? encoder num? loss? end structure
train_id = 's51_leaky_3block_wfl_gp_HE'

train_log_path = r'./' + train_id
if not os.path.exists(train_log_path):
    os.makedirs(train_log_path)

best_acc_model_path = os.path.join(train_log_path, 'bestAcc.h5')
best_loss_model_path = os.path.join(train_log_path, 'bestLoss.h5')
test_ba_npz_path = os.path.join(train_log_path, 'ba_pred.npz')
test_bl_npz_path = os.path.join(train_log_path, 'bl_pred.npz')

label_true_path = os.path.join(train_log_path, 'all_true.npy')
label_pred_path = os.path.join(train_log_path, 'all_pred.npy')
sample_split_ = os.path.join(train_log_path, 'sample_split.npz')


df_path = r'../data/processed_mutations.dataset'
df = pd.read_pickle(df_path)
df = df[~(df['label'] == 4)]


aaDict = {'0':0, 'D':1, 'S':2, 'Q':3, 'K':4,
          'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
          'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
          'A':15, 'V':16, 'E':17, 'Y':18, 'M':19, 'C':20}

max_len = 1024
window_len = 51

mut0_c = [[aaDict[x] for x in a] for a in df['mut0_51']]
mut1_c = [[aaDict[x] for x in a] for a in df['mut1_51']]
par0_c = [[aaDict[x] for x in a] for a in df['par0']]

mut0_c = keras.preprocessing.sequence.pad_sequences(mut0_c, maxlen=window_len, padding='post')
mut1_c = keras.preprocessing.sequence.pad_sequences(mut1_c, maxlen=window_len, padding='post')
par0_c = keras.preprocessing.sequence.pad_sequences(par0_c, maxlen=max_len, padding='post')

pssm_win_mut0 = df['pssm_win_mut0'].values
pssm_win_mut0 = np.stack(pssm_win_mut0, axis=0).astype('float32')
pssm_win_mut1 = df['pssm_win_mut1'].values
pssm_win_mut1 = np.stack(pssm_win_mut1, axis=0).astype('float32')
pssm_par0 = df['pssm_par0'].values
pssm_par0 = [x[:1024, :].astype('float32') for x in pssm_par0]
pssm_par0 = np.stack(pssm_par0, axis=0).astype('float32')

label = df['label'].values


all_index = np.arange(label.shape[0])
print('all_index num: {}'.format(all_index.shape))

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
skf_split = []

for train_index, test_index in skf.split(all_index, label):
    skf_split.append((train_index, test_index))
    
label = to_categorical(label, num_classes=4)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


fold_count = 0
model_loss = []
model_acc = []
model_top2acc = []
model_m = []
all_true = []
all_pred = []


for train_index, test_index in skf_split:
    
    y_train = label[train_index]
    x_da_index = train_index[np.where(y_train.argmax(axis=-1) != 0)]
    
    x_train = [np.r_[mut0_c[train_index], mut1_c[x_da_index], mut0_c[train_index]], 
               np.r_[mut1_c[train_index], mut0_c[x_da_index], mut0_c[train_index]],
               np.r_[par0_c[train_index], par0_c[x_da_index], par0_c[train_index]],
               np.r_[pssm_win_mut0[train_index], pssm_win_mut1[x_da_index], pssm_win_mut0[train_index]],
               np.r_[pssm_win_mut1[train_index], pssm_win_mut0[x_da_index], pssm_win_mut0[train_index]],
               np.r_[pssm_par0[train_index], pssm_par0[x_da_index], pssm_par0[train_index]]]
    
    # y_train WITH augmentation
    y_train_da = y_train[np.where(y_train.argmax(axis=-1) != 0)]
    y_train_no = np.zeros(y_train.shape)
    y_train_no[:, 2] = 1
    y_train = np.r_[y_train, y_train_da[:, [0, 3, 2, 1]], y_train_no]
    
    # do shuffle before training
    shuffle_index = np.arange(x_train[0].shape[0])
    np.random.shuffle(shuffle_index)
    
    x_train = [x_train[0][shuffle_index], x_train[1][shuffle_index], x_train[2][shuffle_index], 
               x_train[3][shuffle_index], x_train[4][shuffle_index], x_train[5][shuffle_index]]
    y_train = y_train[shuffle_index]
    
    vali_index, test_index, vali_label, test_label = train_test_split(test_index, label[test_index], test_size=0.5, 
                                                                      stratify=label[test_index], random_state=0)
    
    x_vali = [mut0_c[vali_index], mut1_c[vali_index], par0_c[vali_index],
              pssm_win_mut0[vali_index], pssm_win_mut1[vali_index], pssm_par0[vali_index]]
    y_vali = label[vali_index]
    
    x_test = [mut0_c[test_index], mut1_c[test_index], par0_c[test_index],
              pssm_win_mut0[test_index], pssm_win_mut1[test_index], pssm_par0[test_index]]
    y_test = label[test_index]
    print('x_train with data augmentation shape: {}'.format(x_train[0].shape))
    print('x_vali shape: {}'.format(x_vali[0].shape))
    print('x_test shape: {}'.format(x_test[0].shape))

    K.clear_session()
    model = build_model()
    if fold_count == 0:
        model.summary()
    
    print('------fold {}-------'.format(str(fold_count)))

    adam = optimizers.Adam(learning_rate=0.0002)

    model.compile(adam, loss=categorical_focal_loss(alpha=[.25, .25, .1, .25], gamma=2.), 
                  metrics=['acc', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2acc')])

    callback = [keras.callbacks.ModelCheckpoint(best_acc_model_path + str(fold_count), monitor='val_acc',
                                                save_best_only=True, save_weights_only=True),
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=12, verbose=0, mode='auto')]

    history = model.fit(
        x_train, y_train, batch_size=64, epochs=150, verbose=2, callbacks=callback, validation_data=(x_vali, y_vali))

    model.load_weights(best_acc_model_path + str(fold_count))
    results = model.evaluate(x_test, y_test, verbose=0)
    results = dict(zip(model.metrics_names, results))
    
    model_loss.append(results['loss'])
    model_acc.append(results['acc'])
    model_top2acc.append(results['top2acc'])
    
    y_pred = model.predict(x_test, verbose=0)
    y_true = y_test
    np.savez(test_ba_npz_path + str(fold_count), y_true=y_true, y_pred=y_pred)
    
    np.savez(sample_split_ + str(fold_count), train=train_index, vali=vali_index, test=test_index)
    
    model_m.append(model_metrics(y_test.argmax(axis=-1), y_pred.argmax(axis=-1)))
    all_true.append(y_true)
    all_pred.append(y_pred)
    
    y_pred, y_true = evaluate_model(model, x_test, y_test)
    fold_count += 1

print('model loss: {} , std {}'.format(np.mean(model_loss, axis=0), np.std(model_loss, axis=0)))
print('model ACC: {} , std {}'.format(np.mean(model_acc, axis=0), np.std(model_acc, axis=0)))
print('model metrics: \nprecision\trecall\tf1\tmcc_score\taccuracy\n{} \n std: \n {}'.format(np.mean(model_m, axis=0), np.std(model_m, axis=0)))


all_true = np.array(all_true)
all_pred = np.array(all_pred)

np.save(label_true_path, all_true)
np.save(label_pred_path, all_pred)




