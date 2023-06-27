import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

stage = 2
#dataset = pd.read_pickle('interface_vis_dataset/partner_stage_' + str(stage) + '_inter_top5att_allset.dataset')
dataset = pd.read_csv('interface_vis_dataset/partner_stage_' + str(stage) + '_inter_top5att_allset.csv')
print(dataset)

list_a = []
list_b = []
head = 'no_1'
for i in range(dataset.shape[0]):
    #print(dataset[i])
    if(dataset['head'][i] == head and dataset['is_interface'][i] == 'not_interface'):
        list_a.append(dataset['attention_output'][i])
    elif(dataset['head'][i] == head and dataset['is_interface'][i] == 'interface'):
        list_b.append(dataset['attention_output'][i])
'''
fig, ax = plt.subplots(figsize=(8,6))
color_list = ['#96CAC1', '#FFDEAD']
flierprops = dict(markerfacecolor='0', markersize=0.01,
                  linestyle='none')
sns.boxplot(x = "head", 
               y = "attention_output", 
               hue = "is_interface", 
               data = dataset, 
               order = ['no_1','no_2','no_3','no_4'], 
               showfliers = False,
               #notch=True,
               palette = color_list,
               #fliersize=20
               )
#ax.get_legend().remove()
#plt.show()
plt.savefig('via_att_plots_new/box_partner_stage_' + str(stage) + '_top5att_allset.png')
'''
a = np.array(list_a)
b = np.array(list_b)

#check = stats.levene(a, b)
#print(check) # p<0.05 set False

t, p = stats.mannwhitneyu(a, b) #Mann-Whitney ranksum test
print(p) #1.5473618462869507e-86
print(len(a))
print(len(b))
# stage2_no_1: 2.13e-27 not/is = 18661/1594  20,255
# stage2_no_2: 7.14e-48 not/is = 18627/1628   20,255
# stage2_no_3: 2.54e-09 not/is = 18,872/1383  20,255
# stage2_no_4: 2.06e-50 not/is = 17,487/2768  20,255 
