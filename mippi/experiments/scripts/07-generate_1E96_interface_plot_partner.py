import os
import numpy as np

import matplotlib.pyplot as plt  
from matplotlib.pyplot import MultipleLocator

# partner chain: 1E96:B/P19878
# mutation chain: 1E96:A/P63000
# interface region:
# P19878[3,36,38,66-69,102,104,106-108,111-112]	P63000[21,24-31,33,36,40-41,160-162]

index = 14 # 1E96
head = 1 # which head
'''
# "1-7, 9-15, 21-46, 50-58, 60-78, 86-111, 129-152, 159-176" 2 residues
interface_region = [(0.8,3.2), (4.8,5.2), (10.8,11.2),(12.8,13.2), (22.8,27.2),(28.8,29.2),(30.8,44.2),(51.8,52,2),(53.8,54.2), 
                    (55.8,56.2), (61.8,67.2),(69.8,70.2) ,(72.8,74.2), (75.8,76.2),(87.8,88.2),(90.8,92.2),(93.8,94.2),
                    (97.8,99.2), (101.8,103.2),(105.8,107.2),(120.8,121.2), (130.8,134.2),
                    (138.8,140.2),(142.8,144.2),(146.8,148.2),(149.8,150.2),(160.8,164.2),(165.8,167.2),
                    (169.8,171.2),(173.8,174.2)]
                    '''
interface_region = [(2.6,3.4),(35.6,36.4), (65.6,69.4),(101.6,102.4), (103.6,104.4),(105.6,108.4),(110.6,112.4)]

dataset1 = np.load('case_study/case_study_stage_0_dataset.npy', allow_pickle=True)
dataset2 = np.load('case_study/case_study_stage_2_dataset.npy', allow_pickle=True)
print(dataset1.shape)
print(dataset1[index])

pdb_name = dataset1[index][0][0:4]
chain_name = dataset1[index][0][-1]
print([pdb_name, chain_name])
chain_length = int(dataset1[index][1])
print(chain_length)

min_att = min(dataset1[index][2][head])
max_att = max(dataset1[index][2][head])
print([min_att, max_att])

print(len(dataset1[index][2][head]))

ini_layer_para = np.load('case_study/1E96_iniweight_stage_dataset.npy')
first_layer_para = dataset1[index][2][head][0:185]
last_layer_para = dataset2[index][2][head][0:185]

x_major_locator=MultipleLocator(10)

plt.figure(figsize=(15, 5))
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(ini_layer_para,color="#E49645",label='embedding layer (input of attention)')
#plt.plot(first_layer_para,'g--',label='type2')
plt.plot(last_layer_para, linewidth=1.1, color='#202072',label='the last self-attention layer')
plt.plot(np.array([80]*185),'--', color='#8DCEBD')

for region in interface_region:
    ax.axvspan(region[0], region[1], alpha=0.35, color='#3CB371')
plt.legend()
plt.savefig('case_partner_plot.png',dpi=300)
plt.show()

