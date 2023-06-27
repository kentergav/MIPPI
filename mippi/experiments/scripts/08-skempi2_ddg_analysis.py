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
df_path = r'skempi2_via_att_plots_new/ddg_dataset.pickle'
df = pd.read_pickle(df_path)
df.reset_index()
print(df.shape)

import matplotlib.pyplot as plt
import seaborn as sns

'''
plt.figure(figsize=(10, 8), dpi=300)
pred_list = ['disrupting', 'decreasing', 'no effect', 'increasing']
#color_list = ['#8F6798', '#83A4BB', '#86D4AF', '#FFF17C']
color_list = ['#FFF17C', '#86D4AF', '#83A4BB', '#8F6798']
sns.boxplot(y="cv5_class", x="ddg", data=df, showfliers = False, palette=color_list, notch=True, orient="h")
plt.yticks([0, 1, 2, 3], pred_list, fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('prediction class', fontsize=20, fontweight='bold')
plt.xlabel('ddG', fontsize=20, fontweight='bold')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('ddg_ori.jpg', dpi=300)

png1 = io.BytesIO()
plt.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("./skempi2_via_att_plots_new/ddg_ori_tif.tiff", dpi=png2.info['dpi'])
png1.close()
'''



'''
import joypy
plt.figure(figsize=(10, 8), dpi=300)
labels = ['disrupting', 'decreasing', 'no effect', 'increasing']
color_list = ['#FFF17C', '#86D4AF', '#83A4BB', '#8F6798']
#for i in range(4):
    #sns.kdeplot(df[df['cv5_class'] == i]['ddg'], shade=True, label=labels[i])
fig,axs = joypy.joyplot(df, 
                        by='cv5_class', 
                        column='ddg', 
                        labels=labels, 
                        overlap=3,
                        color=color_list,
                        linewidth=1)
#plt.legend(fontsize='x-large', title_fontsize='20')
#ax = plt.axes()
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
plt.xlabel('ddG', fontsize=20)
plt.ylabel('Density', fontsize=20)
#plt.savefig('density_distribution.jpg', dpi=300)

png1 = io.BytesIO()
plt.savefig(png1, format="png", dpi=300)

# Load this image into PIL
png2 = Image.open(png1)
print(png2.info)

# Save as TIFF
print(png2.info['dpi'])
png2.save("./skempi2_via_att_plots_new/density_distribution_tif.tiff", dpi=(300, 300))
png1.close()
'''


'''
counts = np.array([8, 8, 32, 6])
counts = counts / sum(counts)
counts_ori = np.array([5452, 3920, 6000, 1079])
counts_ori = counts_ori / sum(counts_ori)
index = ['disrupting', 'decreasing', 'no effect', 'increasing']
df_ca_bar = pd.DataFrame({'causing entries': counts, 'whole train set': counts_ori}, index=index)
plt.figure(dpi=300)
color_list = ['#34626C', '#CFD3CE']
from matplotlib.colors import ListedColormap
my_gsb = ListedColormap(color_list)
ax = df_ca_bar.plot.bar(rot=0, figsize=(10, 7), ax=plt.gca(), colormap = my_gsb)

for p in ax.patches:
    ax.annotate(str(p.get_height().round(2)), (p.get_x() * 1 + 0.12, p.get_height() * 1 + 0.005), fontsize=15, horizontalalignment='center')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('prediction class', fontsize=20, fontweight='bold')
plt.ylabel('proportion', fontsize=20, fontweight='bold')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize='x-large', title_fontsize='20')

png1 = io.BytesIO()
plt.savefig(png1, format="png", dpi=300)

# Load this image into PIL
png2 = Image.open(png1)
print(png2.info)

# Save as TIFF
png2.save("./skempi2_via_att_plots_new/causing_items.tiff", dpi=png2.info['dpi'])
png1.close()
'''


'''
df_con_b_ =  pd.DataFrame(np.array([[77, 8, 7], [365, 66, 6]]), index=['No agree', 'With agree'],
                         columns=pd.Index(['mild', 'middle', 'bad'], name='conflict type'))
plt.figure(dpi=300)
color_list = ['#F0F0F9', '#9BA4B4', '#15284F']
from matplotlib.colors import ListedColormap
my_gsb = ListedColormap(color_list)
df_con_b_.plot.bar(stacked=True, figsize=(8, 5), rot=0, ax=plt.gca(), colormap=my_gsb)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('prediction distribution', fontsize=20, fontweight='bold')
plt.ylabel('pair number', fontsize=20, fontweight='bold')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize='x-large', title_fontsize='20')

png1 = io.BytesIO()
plt.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)
print(png2.info)

# Save as TIFF
png2.save("./skempi2_via_att_plots_new/conflict_items.tiff", dpi=png2.info['dpi'])
png1.close()
'''


from scipy import stats
'''
cv5class = 3
print(df[df['cv5_class'] == cv5class])
x = []
y = []
for item in df[df['cv5_class'] == cv5class]['cv5_score']:
    x.append(item)
for item in df[df['cv5_class'] == cv5class]['ddg']:
    y.append(item)
new_df = pd.DataFrame(columns=('cv5_score','ddg'))
for i in range(20):
    score_range = [0.05 * i, 0.05 * (i + 1)]
    sum = 0
    count = 0
    for j in range(len(x)):
        if(x[j] >= score_range[0] and x[j] < score_range[1]):
            sum += y[j]
            count += 1
    if(count == 0):
        continue
    else:
        avg = float(sum)/count
        new_df = new_df.append(pd.DataFrame({'cv5_score':[score_range[0]],'ddg':[avg]}),ignore_index=True)
slope, intercept, r_value, p_value, std_err = stats.linregress(new_df['cv5_score'], new_df['ddg'])
print(new_df)
print(r_value, p_value)
print("R-squared: %f" % r_value**2)
'''


cv5class = 3
from scipy import stats
x, y = df[df['cv5_class'] == cv5class]['cv5_score'], df[df['cv5_class'] == cv5class]['ddg']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(r_value, p_value)
print("R-squared: %f" % r_value**2)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)
df_a = df[(df['cv5_class'] == cv5class)]
#print(df_a)
g = plt.figure(figsize=(10, 10), dpi=300)
g = sns.JointGrid(data=df_a, x='cv5_score', y='ddg', height=5, xlim=(0.2, 0.9), ylim=(-8, 6))
#g = sns.JointGrid(data=df_a, x='cv5_score', y='ddg', height=5, xlim=(0.2, 0.9), ylim=(-4, 8))
g = g.plot_joint(sns.regplot, 
                 #color="xkcd:muted blue",
                 scatter_kws={"color": "#D8BFD8",'s':5}, 
                 line_kws={"color": "#8F6798"})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('prediction confidence score', fontsize=20)
plt.ylabel('ddG', fontsize=20)
# plt.tight_layout()
g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="#8F6798")
#increasing
g.ax_joint.text(0.5, 2.3, 'r = -0.52, p = 1.9e-10', fontstyle='italic', fontsize=20)
#no effect
#g.ax_joint.text(0.6, 6, 'r = -0.13, p = 2e-4', fontstyle='italic', fontsize=20)
# decreasing
#g.ax_joint.text(0.6, 6, 'r = 0.01, p = 0.88', fontstyle='italic', fontsize=20)
# disrupting
#g.ax_joint.text(0.5, 7, 'r = 0.18, p = 1.7e-12', fontstyle='italic', fontsize=20)

g.fig.set_size_inches(8,8)

# plt.title('decreasing prediction ddg')
# plt.savefig('skempi2_increasing_withtest_copy1_test.jpg', dpi=300)

png1 = io.BytesIO()
plt.savefig(png1, format="png", dpi=300)

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
print(png2.info['dpi'])
png2.save("./skempi2_via_att_plots_new/skempi2_increasing_withtest_tif.tiff", dpi=png2.info['dpi'])
#png2.save("./skempi2_via_att_plots_new/skempi2_noeffect_withtest_tif.tiff", dpi=png2.info['dpi'])
#png2.save("./skempi2_via_att_plots_new/skempi2_decreasing_withtest_tif.tiff", dpi=png2.info['dpi'])
#png2.save("./skempi2_via_att_plots_new/skempi2_disrupting_withtest_tif.tiff", dpi=png2.info['dpi'])
png1.close()
