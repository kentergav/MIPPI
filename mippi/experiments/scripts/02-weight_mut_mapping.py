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
print(df.head(8))
'''
with open('H_sapiens_interfacesHQ_cleaning_pdb.txt') as pdb_all:
    lines = pdb_all.readlines()
    att51_ori_sum_0 = np.load('att_weight/att51_ori/att51_ori_weight_0.npy') #(16505, 4, 51)
    att51_mut_sum_0 = np.load('att_weight/att51_mut/att51_mut_weight_0.npy') #(16505, 4, 51)         
    att51_ori_sum_1 = np.load('att_weight/att51_ori/att51_ori_weight_1.npy') #(16505, 4, 51)
    att51_mut_sum_1 = np.load('att_weight/att51_mut/att51_mut_weight_1.npy') #(16505, 4, 51)     
    att51_ori_sum_2 = np.load('att_weight/att51_ori/att51_ori_weight_2.npy') #(16505, 4, 51)
    att51_mut_sum_2 = np.load('att_weight/att51_mut/att51_mut_weight_2.npy') #(16505, 4, 51)     
    ori_pdb_dataset_0 = []      
    mut_pdb_dataset_0 = []
    ori_pdb_dataset_1 = []      
    mut_pdb_dataset_1 = []
    ori_pdb_dataset_2 = []      
    mut_pdb_dataset_2 = []    
    for i in range(df.shape[0]):
        for line in lines:
            pairs = [line.split()[0], line.split()[1]]        
            if([df['mutAC'][i], df['parAC'][i]] == pairs):
                #print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i]])
                ori_pdb_dataset_0.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_0[i][0], att51_ori_sum_0[i][1], att51_ori_sum_0[i][2], att51_ori_sum_0[i][3]])     
                mut_pdb_dataset_0.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_0[i][0], att51_mut_sum_0[i][1], att51_mut_sum_0[i][2], att51_mut_sum_0[i][3]])      
                ori_pdb_dataset_1.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_1[i][0], att51_ori_sum_1[i][1], att51_ori_sum_1[i][2], att51_ori_sum_1[i][3]])     
                mut_pdb_dataset_1.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_1[i][0], att51_mut_sum_1[i][1], att51_mut_sum_1[i][2], att51_mut_sum_1[i][3]])  
                ori_pdb_dataset_2.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_2[i][0], att51_ori_sum_2[i][1], att51_ori_sum_2[i][2], att51_ori_sum_2[i][3]])     
                mut_pdb_dataset_2.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_2[i][0], att51_mut_sum_2[i][1], att51_mut_sum_2[i][2], att51_mut_sum_2[i][3]])                  
                       
            elif([df['parAC'][i], df['mutAC'][i]] == pairs):
                #print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i]])
                ori_pdb_dataset_0.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_0[i][0], att51_ori_sum_0[i][1], att51_ori_sum_0[i][2], att51_ori_sum_0[i][3]])     
                mut_pdb_dataset_0.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_0[i][0], att51_mut_sum_0[i][1], att51_mut_sum_0[i][2], att51_mut_sum_0[i][3]])      
                ori_pdb_dataset_1.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_1[i][0], att51_ori_sum_1[i][1], att51_ori_sum_1[i][2], att51_ori_sum_1[i][3]])     
                mut_pdb_dataset_1.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_1[i][0], att51_mut_sum_1[i][1], att51_mut_sum_1[i][2], att51_mut_sum_1[i][3]])  
                ori_pdb_dataset_2.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_2[i][0], att51_ori_sum_2[i][1], att51_ori_sum_2[i][2], att51_ori_sum_2[i][3]])     
                mut_pdb_dataset_2.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_2[i][0], att51_mut_sum_2[i][1], att51_mut_sum_2[i][2], att51_mut_sum_2[i][3]])                     
    
    ori_pdb_dataset_0_np = np.array(ori_pdb_dataset_0)
    print(ori_pdb_dataset_0_np.shape) #(868, 8)
    np.save('interface_att_dataset/ori_pdb_stage_0.npy', ori_pdb_dataset_0_np)
    mut_pdb_dataset_0_np = np.array(mut_pdb_dataset_0)
    print(mut_pdb_dataset_0_np.shape) #(868, 8)
    np.save('interface_att_dataset/mut_pdb_stage_0.npy', mut_pdb_dataset_0_np)      
    ori_pdb_dataset_1_np = np.array(ori_pdb_dataset_1)
    print(ori_pdb_dataset_1_np.shape) #(868, 8)
    np.save('interface_att_dataset/ori_pdb_stage_1.npy', ori_pdb_dataset_1_np)
    mut_pdb_dataset_1_np = np.array(mut_pdb_dataset_1)
    print(mut_pdb_dataset_1_np.shape) #(868, 8)
    np.save('interface_att_dataset/mut_pdb_stage_1.npy', mut_pdb_dataset_1_np) 
    ori_pdb_dataset_2_np = np.array(ori_pdb_dataset_2)
    print(ori_pdb_dataset_2_np.shape) #(868, 8)
    np.save('interface_att_dataset/ori_pdb_stage_2.npy', ori_pdb_dataset_2_np)
    mut_pdb_dataset_2_np = np.array(mut_pdb_dataset_2)
    print(mut_pdb_dataset_2_np.shape) #(868, 8)
    np.save('interface_att_dataset/mut_pdb_stage_2.npy', mut_pdb_dataset_2_np)                            
        '''
        
with open('H_sapiens_interfacesHQ_cleaning_all.txt') as mapping_all:
    lines = mapping_all.readlines()
    att51_ori_sum_0 = np.load('att_weight/att51_ori/att51_ori_weight_0.npy') #(16505, 4, 51)
    att51_mut_sum_0 = np.load('att_weight/att51_mut/att51_mut_weight_0.npy') #(16505, 4, 51)         
    att51_ori_sum_1 = np.load('att_weight/att51_ori/att51_ori_weight_1.npy') #(16505, 4, 51)
    att51_mut_sum_1 = np.load('att_weight/att51_mut/att51_mut_weight_1.npy') #(16505, 4, 51)     
    att51_ori_sum_2 = np.load('att_weight/att51_ori/att51_ori_weight_2.npy') #(16505, 4, 51)
    att51_mut_sum_2 = np.load('att_weight/att51_mut/att51_mut_weight_2.npy') #(16505, 4, 51)     
    ori_pdb_dataset_0 = []      
    mut_pdb_dataset_0 = []
    ori_pdb_dataset_1 = []      
    mut_pdb_dataset_1 = []
    ori_pdb_dataset_2 = []      
    mut_pdb_dataset_2 = []    
    for i in range(df.shape[0]):
        for line in lines:
            pairs = [line.split()[0], line.split()[1]]        
            if([df['mutAC'][i], df['parAC'][i]] == pairs):
                print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i]])
                ori_pdb_dataset_0.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_0[i][0], att51_ori_sum_0[i][1], att51_ori_sum_0[i][2], att51_ori_sum_0[i][3]])     
                mut_pdb_dataset_0.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_0[i][0], att51_mut_sum_0[i][1], att51_mut_sum_0[i][2], att51_mut_sum_0[i][3]])      
                ori_pdb_dataset_1.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_1[i][0], att51_ori_sum_1[i][1], att51_ori_sum_1[i][2], att51_ori_sum_1[i][3]])     
                mut_pdb_dataset_1.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_1[i][0], att51_mut_sum_1[i][1], att51_mut_sum_1[i][2], att51_mut_sum_1[i][3]])  
                ori_pdb_dataset_2.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_2[i][0], att51_ori_sum_2[i][1], att51_ori_sum_2[i][2], att51_ori_sum_2[i][3]])     
                mut_pdb_dataset_2.append([0, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_2[i][0], att51_mut_sum_2[i][1], att51_mut_sum_2[i][2], att51_mut_sum_2[i][3]])                  
                       
            elif([df['parAC'][i], df['mutAC'][i]] == pairs):
                print([df['mutAC'][i], df['parAC'][i], df['Feature range(s)'][i]])
                ori_pdb_dataset_0.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_0[i][0], att51_ori_sum_0[i][1], att51_ori_sum_0[i][2], att51_ori_sum_0[i][3]])     
                mut_pdb_dataset_0.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_0[i][0], att51_mut_sum_0[i][1], att51_mut_sum_0[i][2], att51_mut_sum_0[i][3]])      
                ori_pdb_dataset_1.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_1[i][0], att51_ori_sum_1[i][1], att51_ori_sum_1[i][2], att51_ori_sum_1[i][3]])     
                mut_pdb_dataset_1.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_1[i][0], att51_mut_sum_1[i][1], att51_mut_sum_1[i][2], att51_mut_sum_1[i][3]])  
                ori_pdb_dataset_2.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_ori_sum_2[i][0], att51_ori_sum_2[i][1], att51_ori_sum_2[i][2], att51_ori_sum_2[i][3]])     
                mut_pdb_dataset_2.append([1, df['label'][i], line.split(), df['Feature range(s)'][i][0].split('-')[0], 
                                   att51_mut_sum_2[i][0], att51_mut_sum_2[i][1], att51_mut_sum_2[i][2], att51_mut_sum_2[i][3]])                     
    
    ori_pdb_dataset_0_np = np.array(ori_pdb_dataset_0)
    print(ori_pdb_dataset_0_np.shape) 
    np.save('interface_att_dataset/ori_all_stage_0.npy', ori_pdb_dataset_0_np)
    mut_pdb_dataset_0_np = np.array(mut_pdb_dataset_0)
    print(mut_pdb_dataset_0_np.shape) 
    np.save('interface_att_dataset/mut_all_stage_0.npy', mut_pdb_dataset_0_np)      
    ori_pdb_dataset_1_np = np.array(ori_pdb_dataset_1)
    print(ori_pdb_dataset_1_np.shape) 
    np.save('interface_att_dataset/ori_all_stage_1.npy', ori_pdb_dataset_1_np)
    mut_pdb_dataset_1_np = np.array(mut_pdb_dataset_1)
    print(mut_pdb_dataset_1_np.shape) 
    np.save('interface_att_dataset/mut_all_stage_1.npy', mut_pdb_dataset_1_np) 
    ori_pdb_dataset_2_np = np.array(ori_pdb_dataset_2)
    print(ori_pdb_dataset_2_np.shape) 
    np.save('interface_att_dataset/ori_all_stage_2.npy', ori_pdb_dataset_2_np)
    mut_pdb_dataset_2_np = np.array(mut_pdb_dataset_2)
    print(mut_pdb_dataset_2_np.shape) 
    np.save('interface_att_dataset/mut_all_stage_2.npy', mut_pdb_dataset_2_np)      
        
        
                       
            