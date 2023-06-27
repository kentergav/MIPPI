import numpy as np
import os
import sys
import pandas as pd

np.random.seed(0)

df_path = r'../../data/processed_mutations.dataset'
# df_path = r'../../data/skempi2_window_with_pssm.dataset'
df = pd.read_pickle(df_path)
pd.set_option('display.max_columns', None)
print(df.columns)
print(df.head(2))

proteins_pair = df['partners'].values
#print(proteins_pair)

with open('H_sapiens_interfacesHQ_cleaning_all.txt') as all_interface:
    all_lines = all_interface.readlines()
with open('H_sapiens_interfacesHQ_cleaning_pdb.txt') as pdb_interface:
    pdb_lines = pdb_interface.readlines()
all_interface_pairs = []
pdb_interface_pairs = []
for line in all_lines:
    all_interface_pairs.append([line.split()[0], line.split()[1]])
for line in pdb_lines:
    pdb_interface_pairs.append([line.split()[0], line.split()[1]])
#print(interface_pairs)


interface_mapping_all = []
for list_item in proteins_pair:
    if([list_item[0], list_item[1]] in all_interface_pairs):
        #print([list_item[0], list_item[1]])
        interface_mapping_all.append([list_item[0], list_item[1]])
    elif([list_item[1], list_item[0]] in all_interface_pairs):
        #print([list_item[1], list_item[0]])
        interface_mapping_all.append([list_item[1], list_item[0]])
#interface_mapping_all_new = list(set(interface_mapping_all))
np.save('interface_mapping_all.npy', interface_mapping_all)

interface_mapping_pdb = []
for list_item in proteins_pair:
    if([list_item[0], list_item[1]] in pdb_interface_pairs):
        #print([list_item[0], list_item[1]])
        interface_mapping_pdb.append([list_item[0], list_item[1]])
    elif([list_item[1], list_item[0]] in pdb_interface_pairs):
        #print([list_item[1], list_item[0]])
        interface_mapping_pdb.append([list_item[1], list_item[0]])
#interface_mapping_pdb_new = list(set(interface_mapping_pdb))
np.save('interface_mapping_pdb.npy', interface_mapping_pdb)

'''
interface_mapping_all = np.load('interface_mapping_all.npy')
with open('H_sapiens_interfacesHQ_mapping_all.txt', 'w+') as file_mapping_all:
    for line in all_lines:
        if([line.split()[0], line.split()[1]] in interface_mapping_all):
            #print([line.split()[0], line.split()[1]])
            file_mapping_all.write(line)

interface_mapping_pdb = np.load('interface_mapping_pdb.npy')
with open('H_sapiens_interfacesHQ_mapping_pdb.txt', 'w+') as file_mapping_pdb:
    for line in pdb_lines:
        if([line.split()[0], line.split()[1]] in interface_mapping_pdb):
            file_mapping_pdb.write(line)
'''
