import numpy as np
import os
import sys
import pandas as pd

np.random.seed(0)

df_path = r'../../data/processed_mutations.dataset'
# df_path = r'../../data/skempi2_window_with_pssm.dataset'
df = pd.read_pickle(df_path)
pd.set_option('display.max_columns', None)
print(df.shape)
print(df.columns)
print(df.head(4))

with open('H_sapiens_interfacesHQ_cleaning_all.txt') as mapping_all:
    lines = mapping_all.readlines()
    partner_all_dataset = []
    
    for i in range(df.shape[0]):
        for line in lines:
            pairs = [line.split()[0], line.split()[1]]        
            if([df['mutAC'][i], df['parAC'][i]] == pairs):
                print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i], len(df['par0'][i]), i])
                partner_all_dataset.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], len(df['par0'][i]), i])                      
                       
            elif([df['parAC'][i], df['mutAC'][i]] == pairs):
                print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i], len(df['par0'][i]), i])
                partner_all_dataset.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], len(df['par0'][i]), i])      

partner_all_dataset_np = np.array(partner_all_dataset)
print(partner_all_dataset_np.shape) #(4051, 6)
np.save('interface_att_dataset/partner_all_dataset.npy', partner_all_dataset_np)
        
