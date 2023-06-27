import pandas as pd
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import scipy.stats as stats
from mippiNetbuild_att import *

pd.set_option('display.max_columns', None)

'''
df = pd.read_csv(r'../../data/psymukb/after_predict_via_att/psymukb_biogrid_processed_via_att.csv')
df_pred = pd.read_csv(r'../../data/psymukb/after_predict_via_att/after_predicted_brief_via_att.csv')
master_table = pd.read_csv(r'../../data/psymukb/processed_psymukb.csv')

print('{}\n{}\n{}'.format(df.shape, df_pred.shape, master_table.shape))

master_table['var_abbr'] = master_table['refseq_mrna'] + '_' + master_table['protein_ori'] + master_table['protein_pos'].astype(str) + master_table['protein_mut']
print(df)
print(df_pred)
df_plus = pd.concat([df, df_pred], axis=1)
df_plus = df_plus.loc[:, ~df_plus.columns.duplicated()]
print(df_plus.shape)
df_plus.reset_index(drop=True, inplace=True)
master_table.reset_index(drop=True, inplace=True)
# temp PPI unit try
df_plus = df_plus[~(df_plus['PrimaryPhenotype'] == 'Uncharacterized (Mixed healthy control)')]
df_plus['sample_type'] = 'case'
# df_plus.loc[df_plus[df_plus['PrimaryPhenotype'] == 'Uncharacterized (Mixed healthy control)'].index, 'sample_type'] = 'control'
df_plus.loc[df_plus[df_plus['PrimaryPhenotype'] == 'Sibling Control'].index, 'sample_type'] = 'control'

df_plus['pred_class'] = df_plus['pred_class'].replace(0, 'disrupting')
df_plus['pred_class'] = df_plus['pred_class'].replace(1, 'decreasing')
df_plus['pred_class'] = df_plus['pred_class'].replace(2, 'no effect')
df_plus['pred_class'] = df_plus['pred_class'].replace(3, 'increasing')

pred_arr = np.zeros((master_table.shape[0], 4))
pred_re_arr = np.zeros((master_table.shape[0], 4))
for i in tqdm(master_table.index):
    df_tmp = df_plus[df_plus['var_abbr'] == master_table.loc[i, 'var_abbr']]
    pred_arr[i, 0] += sum(df_tmp['pred_class'] == 0)
    pred_arr[i, 1] += sum(df_tmp['pred_class'] == 1)
    pred_arr[i, 2] += sum(df_tmp['pred_class'] == 2)
    pred_arr[i, 3] += sum(df_tmp['pred_class'] == 3)
    pred_re_arr[i, 0] += sum(df_tmp['pred_re_class'] == 0)
    pred_re_arr[i, 1] += sum(df_tmp['pred_re_class'] == 1)
    pred_re_arr[i, 2] += sum(df_tmp['pred_re_class'] == 2)
    pred_re_arr[i, 3] += sum(df_tmp['pred_re_class'] == 3)
    
pred_arr = pred_arr.astype(int)
pred_re_arr = pred_re_arr.astype(int)

master_table['pred_most_common'] = pred_arr.argmax(axis=-1)
master_table['pred_disrupting'] = pred_arr[:, 0]
master_table['pred_decreasing'] = pred_arr[:, 1]
master_table['pred_noeffect'] = pred_arr[:, 2]
master_table['pred_increasing'] = pred_arr[:, 3]
master_table['pred_sum'] = pred_arr.sum(axis=-1)

master_table['pred_re_most_common'] = pred_re_arr.argmax(axis=-1)
master_table['pred_re_disrupting'] = pred_re_arr[:, 0]
master_table['pred_re_decreasing'] = pred_re_arr[:, 1]
master_table['pred_re_noeffect'] = pred_re_arr[:, 2]
master_table['pred_re_increasing'] = pred_re_arr[:, 3]
'''
master_table = pd.read_csv('./psymukb_via_att_plots/psymukb_mippi_done_att.tsv', sep='\t')
print(master_table.shape)
master_table['pred_most_common_type'] = master_table['pred_most_common']
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(0, 'disrupting')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(1, 'decreasing')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(2, 'no effect')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(3, 'increasing')

master_table['sample_type'] = 'case'
master_table.loc[master_table[master_table['PrimaryPhenotype'] == 'Uncharacterized (Mixed healthy control)'].index, 'sample_type'] = 'control'
master_table.loc[master_table[master_table['PrimaryPhenotype'] == 'Sibling Control'].index, 'sample_type'] = 'control'
master_table_original = master_table

master_table = master_table[~(master_table['PrimaryPhenotype'] == 'Uncharacterized (Mixed healthy control)')]


color_list1 = ['#FFF17C', '#86D4AF', '#83A4BB', '#8F6798']
color_list2 = ['#F08080', '#CFD3CE']
color_list3 = ['#AFEEEE', '#CFD3CE']

'''
group1 = [3713, 651, 5902, 651]
group2 = [315, 60, 605, 60]
group_name = ['disrupting', 'decreasing', 'no effect', 'increasing']
adjust_cof = len(group1) * (len(group1) - 1) / 2
for i in range(len(group1) - 1):
    for j in range(i + 1, len(group1)):
        odds, p = stats.fisher_exact([[group1[i], group2[i]], [group1[j], group2[j]]])
        print('{} and {}, OR: {:.3f}, p-value: {:.5f}'.format(group_name[i], group_name[j], odds, p * adjust_cof))
# stats.fisher_exact([[3015, 235], [916, 80]])
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
my_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
plot = sns.countplot(data=master_table, x='pred_most_common_type', ax=ax, order=my_order, hue='sample_type', palette=color_list2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

for p in plot.patches:
    plot.annotate(p.get_height(), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

ax.set_xlabel('most common type of prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('count', fontsize=18, fontweight='bold')

# statistical annotation
x1, x2 = 0, 2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = 6244 + 600, 200, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x1 - 0.2, x1 + 0.2], [y, y], lw=1.5, c=col)
plt.plot([x2 - 0.2, x2 + 0.2], [y, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "OR = 1.21, P-value = 0.057", ha='center', va='bottom', color=col)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_title('')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./psymukb_via_att_plots_new/pred_type_case_control_count_nomixhealthy.tiff", dpi=png2.info['dpi'])
png1.close()
'''

'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
my_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
x, y = 'pred_most_common_type', 'sample_type'
df_1 = master_table.groupby(y)[x].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()

plot = sns.barplot(x=x, y='percent', hue=y, data=df_1, order=my_order, palette=color_list2)
plot.set_ylim(0, 100)
# plot = sns.countplot(data=master_table, x='pred_most_common_type', ax=ax, order=my_order, hue='sample_type')
# plt.rcParams["axes.labelsize"] = 12
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

ax.set_xlabel('most common type of prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=18, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_title('')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./psymukb_via_att_plots_new/pred_type_case_control_percent_nomixhealthy.tiff", dpi=png2.info['dpi'])
png1.close()
'''


'''
master_table.groupby(by=['PrimaryPhenotype'])['pred_most_common'].value_counts(sort=False)
sns.set(font_scale=2, style='white')
fig, ax = plt.subplots(figsize=(18, 11), dpi=300)
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
x, y = 'PrimaryPhenotype', 'pred_most_common_type'
df_2 = master_table.groupby('PrimaryPhenotype').filter(lambda x: len(x) > 100)
df_1 = df_2.groupby(x)[y].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()
df_1 = df_1.sort_values(['percent'], ascending=False).reset_index(drop=True)
print(df_1)
df_1.to_excel('excel_psymukb.xlsx',sheet_name='psymukb')

test_plot = sns.barplot(data=df_1, x=x, y='percent', hue=y, ax=ax, hue_order=hue_order, palette=color_list1)
# for item in test_plot.get_xticklabels():
#     item.set_rotation(90)

ax.set_xlabel('primary phenotype', fontsize=30, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, rotation=30, ha='right')
# ax.set_xticks(fontsize=30)
# ax.set_xticklabels(ax.xticklabels, rotation=40, ha='right')
# plt.legend(fontsize=30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=30).set_title('')
# plt.legend(loc='upper right')
# plt.tight_layout()

test_plot.figure.savefig("./psymukb_via_att_plots_new/pred_type_disease_percent100_nomixhealthy.tiff", dpi=300, bbox_inches='tight')
'''

'''
group1 = [2894, 453, 3484, 346]
group2 = [1029, 235, 2884, 349]
group_name = ['disrupting', 'decreasing', 'no effect', 'increasing']
adjust_cof = len(group1) * (len(group1) - 1) / 2
for i in range(len(group1) - 1):
    for j in range(i + 1, len(group1)):
        odds, p = stats.fisher_exact([[group1[i], group1[j]], [group2[i], group2[j]]])
#         print([[group1[i], group2[i]], [group1[j], group2[j]]])
        print('{} and {}, OR: {:.3f}, p-value: {:.9f}'.format(group_name[i], group_name[j], odds, p * adjust_cof))
        
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
test_plot = sns.countplot(data=master_table[~(master_table['SIFT_pred'] == '.')], x='SIFT_pred', hue='pred_most_common_type', ax=ax, hue_order=hue_order, palette=color_list1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for p in test_plot.patches:
    test_plot.annotate(p.get_height(), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
ax.legend().set_title('')
ax.legend(fontsize='13')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set_xlabel('SIFT prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('count', fontsize=18, fontweight='bold')
test_plot.set(xticklabels=['deleterious', ' tolerated'])


x1, x2 = -0.1, 0.9   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = 3764 + 600, 200, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x1 - 0.2, x1 - 0.2, x1 + 0.2, x1 + 0.2], [y - h * 0.5, y, y, y - h * 0.5], lw=1.5, c=col)
plt.plot([x2 - 0.2, x2 - 0.2, x2 + 0.2, x2 + 0.2], [y - h * 0.5, y, y, y - h * 0.5], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "OR = 2.328, P-value < 1e-8", ha='center', va='bottom', color=col)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.legend(loc='upper right')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./psymukb_via_att_plots_new/pred_type_SIFT_count_nomixhealthy.tiff", dpi=png2.info['dpi'])
png1.close()
'''

'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
x, y = 'SIFT_pred', 'pred_most_common_type'
df_1 = master_table[~(master_table['SIFT_pred'] == '.')]
df_1 = df_1.groupby(x)[y].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()
test_plot = sns.barplot(data=df_1, x=x, y='percent', hue=y, ax=ax, hue_order=hue_order, palette=color_list1)

for p in test_plot.patches:
    test_plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
ax.legend().set_title('')
ax.legend(fontsize='15')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set_xlabel('SIFT prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=18, fontweight='bold')
test_plot.set(xticklabels=['deleterious', ' tolerated'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# # plt.legend(loc='upper right')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./psymukb_via_att_plots_new/pred_type_SIFT_percent_nomixhealthy.tiff", dpi=png2.info['dpi'])
png1.close()
'''



'''
import joypy
tmp_table = master_table[~(master_table['SIFT_score'] == '.')]
print(tmp_table)
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
a = tmp_table[tmp_table['pred_most_common_type'] == 'disrupting'].replace('disrupting', 0) 
b = tmp_table[tmp_table['pred_most_common_type'] == 'decreasing'].replace('decreasing', 1) 
c = tmp_table[tmp_table['pred_most_common_type'] == 'no effect'].replace('no effect', 2) 
d = tmp_table[tmp_table['pred_most_common_type'] == 'increasing'].replace('increasing', 3) 
new_df = pd.concat([a,b,c,d])
new_df['SIFT_converted_rankscore'] = new_df['SIFT_converted_rankscore'].astype(float)
print(new_df['pred_most_common_type'])
print(new_df['SIFT_converted_rankscore'])
'''
'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')

sns.kdeplot(a['SIFT_converted_rankscore'], label='disrupting', ax=ax)
sns.kdeplot(b['SIFT_converted_rankscore'], label='decreasing', ax=ax)
sns.kdeplot(c['SIFT_converted_rankscore'], label='no effect', ax=ax)
sns.kdeplot(d['SIFT_converted_rankscore'], label='increasing', ax=ax)
'''
'''
plt.figure(figsize=(10, 8), dpi=300)
fig,ax = joypy.joyplot(new_df, 
                        by='pred_most_common_type', 
                        column='SIFT_converted_rankscore', 
                        labels=hue_order, 
                        overlap=3,
                        color=color_list1,
                        linewidth=1)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./psymukb_via_att_plots_new/pred_type_distribution_kde_nomixhealthy.tiff", dpi=(300,300))
png1.close()
'''