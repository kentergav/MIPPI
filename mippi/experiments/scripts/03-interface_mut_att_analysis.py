import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#pd.set_option('display.max_rows', None)

mut_stage_0 = np.load('interface_att_dataset/mut_all_stage_2.npy', allow_pickle=True)
#ori_stage_0 = np.load('interface_att_dataset/ori_pdb_stage_0.npy', allow_pickle=True)
print(mut_stage_0[0])

mut_stage_0_inter = pd.DataFrame(columns=('attention_output','is_interface','head'))
for i in range(mut_stage_0.shape[0]):
#for i in range(100):
    interface_range = mut_stage_0[i][2][3 + int(mut_stage_0[i][0])].replace('[','').replace(']','').split(',') #mut
    #interface_range = mut_stage_0[i][2][4 - int(mut_stage_0[i][0])].replace('[','').replace(']','').split(',') #ori
    print(interface_range)
    mut_pos = int(mut_stage_0[i][3])
    print(mut_pos)
    for res in range(mut_pos - 20, mut_pos + 21):
        flag = 0
        for region in interface_range:
            if('-' in region):
                if(res >= int(region.split('-')[0]) - 5 and res <= int(region.split('-')[1]) + 5):
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-4][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_1']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-3][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_2']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-2][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_3']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-1][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_4']}),ignore_index=True)
                    flag = 1
            else:
                if(res >= int(region) - 5 and res <= int(region) + 5):
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-4][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_1']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-3][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_2']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-2][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_3']}),ignore_index=True)
                    mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-1][res - mut_pos + 20]],'is_interface':['interface'],'head':['no_4']}),ignore_index=True)      
                    flag = 1
        if(flag == 0):   
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-4][res - mut_pos + 20]],'is_interface':['not_interface'],'head':['no_1']}),ignore_index=True)
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-3][res - mut_pos + 20]],'is_interface':['not_interface'],'head':['no_2']}),ignore_index=True)
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-2][res - mut_pos + 20]],'is_interface':['not_interface'],'head':['no_3']}),ignore_index=True)
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[mut_stage_0[i][-1][res - mut_pos + 20]],'is_interface':['not_interface'],'head':['no_4']}),ignore_index=True)  
    
print(mut_stage_0_inter.shape) #(833096, 3)
#print(mut_stage_0_inter)

sns.violinplot(x = "head", 
               y = "attention_output", 
               hue = "is_interface", 
               data = mut_stage_0_inter, 
               order = ['no_1','no_2','no_3','no_4'], 
               #scale = 'count', 
               split = True, 
               palette = 'RdBu' 
              )
plt.savefig('via_att_plots_new/mut_stage_2_inter_att_allset.png')
