import os
import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#pd.set_option('display.max_rows', None)

partner_all_dataset_np = np.load('interface_att_dataset/partner_all_dataset.npy', allow_pickle=True)
print(partner_all_dataset_np.shape)
att_path = 'att_weight/att1024_partner/'
#att1024_partner_0_weight_0 = np.load(att_path + 'att1024_partner_0_weight_0.npy', allow_pickle=True)
#print(att1024_partner_0_weight_0.shape)

#for stage in range(3):
stage = 2
#mut_stage_0_inter = pd.read_pickle('interface_vis_dataset/partner_stage_' + str(stage) + '_inter_top5att_allset.dataset')

mut_stage_0_inter = pd.DataFrame(columns=('attention_output','is_interface','head'))
for item in partner_all_dataset_np:
    idx = item[-1]
    seq_length = min(item[-2], 1024)
    print([seq_length, idx])
    weight = np.load(att_path + 'att1024_partner_' + str(idx) + '_weight_' + str(stage) + '.npy', allow_pickle=True)
    #print(weight)
    interface_range = item[2][3 + int(item[0])].replace('[','').replace(']','').split(',') 
    
    # head_1
    #top5_head_1 = np.argpartition(weight[0][:seq_length], -5)[-5:]
    top5_head_1 = [np.argmax(weight[0][:seq_length]),np.argsort(weight[0][:seq_length])[-2],np.argsort(weight[0][:seq_length])[-3],np.argsort(weight[0][:seq_length])[-4],np.argsort(weight[0][:seq_length])[-5]]
    #print(len(top5_head_1))
    #top5_head_1 = [np.argmax(weight[0][:seq_length])]
    for res in top5_head_1:
        flag = 0
        for region in interface_range:
            if('-' in region):
                if(res >= int(region.split('-')[0]) - 1 and res <= int(region.split('-')[1]) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-4][res]],'is_interface':['interface'],'head':['no_1']}),ignore_index=True)
                    flag = 1
            else:
                if(res >= int(region) - 1 and res <= int(region) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-4][res]],'is_interface':['interface'],'head':['no_1']}),ignore_index=True)     
                    flag = 1
        if(flag == 0):   
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-4][res]],'is_interface':['not_interface'],'head':['no_1']}),ignore_index=True)    
        else:
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-4][res]],'is_interface':['interface'],'head':['no_1']}),ignore_index=True)     
    
    # head_2
    #top5_head_2 = np.argpartition(weight[1][:seq_length], -5)[-5:]
    top5_head_2 = [np.argmax(weight[1][:seq_length]),np.argsort(weight[1][:seq_length])[-2],np.argsort(weight[1][:seq_length])[-3],np.argsort(weight[1][:seq_length])[-4],np.argsort(weight[1][:seq_length])[-5]]
    #print(len(top5_head_2))
    #top5_head_2 = [np.argmax(weight[1][:seq_length])]
    for res in top5_head_2:
        flag = 0
        for region in interface_range:
            if('-' in region):
                if(res >= int(region.split('-')[0]) - 1 and res <= int(region.split('-')[1]) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-3][res]],'is_interface':['interface'],'head':['no_2']}),ignore_index=True)
                    flag = 1
            else:
                if(res >= int(region) - 1 and res <= int(region) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-3][res]],'is_interface':['interface'],'head':['no_2']}),ignore_index=True)     
                    flag = 1
        if(flag == 0):   
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-3][res]],'is_interface':['not_interface'],'head':['no_2']}),ignore_index=True)    
        else:
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-3][res]],'is_interface':['interface'],'head':['no_2']}),ignore_index=True)     
    
    # head_3
    #top5_head_3 = np.argpartition(weight[2][:seq_length], -5)[-5:]
    top5_head_3 = [np.argmax(weight[2][:seq_length]),np.argsort(weight[2][:seq_length])[-2],np.argsort(weight[2][:seq_length])[-3],np.argsort(weight[2][:seq_length])[-4],np.argsort(weight[2][:seq_length])[-5]]
    #print(len(top5_head_3))
    #top5_head_3 = [np.argmax(weight[2][:seq_length])]
    for res in top5_head_3:
        flag = 0
        for region in interface_range:
            if('-' in region):
                if(res >= int(region.split('-')[0]) - 1 and res <= int(region.split('-')[1]) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-2][res]],'is_interface':['interface'],'head':['no_3']}),ignore_index=True)
                    flag = 1
            else:
                if(res >= int(region) - 1 and res <= int(region) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-2][res]],'is_interface':['interface'],'head':['no_3']}),ignore_index=True)     
                    flag = 1
        if(flag == 0):   
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-2][res]],'is_interface':['not_interface'],'head':['no_3']}),ignore_index=True)
        else:
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-2][res]],'is_interface':['interface'],'head':['no_3']}),ignore_index=True)
    
    # head_4
    #top5_head_4 = np.argpartition(weight[3][:seq_length], -5)[-5:]
    top5_head_4 = [np.argmax(weight[3][:seq_length]),np.argsort(weight[3][:seq_length])[-2],np.argsort(weight[3][:seq_length])[-3],np.argsort(weight[3][:seq_length])[-4],np.argsort(weight[3][:seq_length])[-5]]
    #print(len(top5_head_4))
    #top5_head_4 = [np.argmax(weight[3][:seq_length])]
    for res in top5_head_4:
        flag = 0
        for region in interface_range:
            if('-' in region):
                if(res >= int(region.split('-')[0]) - 1 and res <= int(region.split('-')[1]) - 1):
                    #mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-1][res]],'is_interface':['interface'],'head':['no_4']}),ignore_index=True)
                    flag = 1
            else:
                if(res >= int(region) - 3 and res <= int(region) + 1):
                    flag = 1
        if(flag == 0):   
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-1][res]],'is_interface':['not_interface'],'head':['no_4']}),ignore_index=True)
        else:
            mut_stage_0_inter = mut_stage_0_inter.append(pd.DataFrame({'attention_output':[weight[-1][res]],'is_interface':['interface'],'head':['no_4']}),ignore_index=True)
print(mut_stage_0_inter.shape)
mut_stage_0_inter.to_csv('interface_vis_dataset/partner_stage_' + str(stage) + '_inter_top5att_allset.csv')
'''
sns.violinplot(x = "head", 
               y = "attention_output", 
               hue = "is_interface", 
               data = mut_stage_0_inter, 
               order = ['no_1','no_2','no_3','no_4'], 
               #scale = 'count', 
               split = False, 
               palette = 'RdBu' 
              )
#plt.show()
plt.savefig('via_att_plots_new/partner_stage_' + str(stage) + '_inter_top5att_allset.png')        
'''