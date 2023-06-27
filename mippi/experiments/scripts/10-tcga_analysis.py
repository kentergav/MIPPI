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

color_list1 = ['#FFF17C', '#86D4AF', '#83A4BB', '#8F6798']
color_list2 = ['#F08080', '#CFD3CE']
color_list3 = ['#AFEEEE', '#CFD3CE']

df = pd.read_csv(r'cancer_ready_for_mippi_via_att.csv')
df_pred = pd.read_csv(r'after_predicted_brief_via_att.csv')
master_table = pd.read_csv(r'processed_cancer_set.csv')

print('{}\n{}\n{}'.format(df.shape, df_pred.shape, master_table.shape))

master_table['var_abbr'] = master_table['refseq_mrna'] + '_' + master_table['protein_ori'] + master_table['protein_pos'].astype(str) + master_table['protein_mut']
df_plus = pd.concat([df, df_pred], axis=1)
df_plus = df_plus.loc[:, ~df_plus.columns.duplicated()]
df_plus.reset_index(drop=True, inplace=True)
master_table.reset_index(drop=True, inplace=True)

df_plus['pred_class'] = df_plus['pred_class'].replace(0, 'disrupting')
df_plus['pred_class'] = df_plus['pred_class'].replace(1, 'decreasing')
df_plus['pred_class'] = df_plus['pred_class'].replace(2, 'no effect')
df_plus['pred_class'] = df_plus['pred_class'].replace(3, 'increasing')

'''
group1 = [126435, 40996, 185499, 34270]
group2 = [15921, 6374, 30338, 5133]
group_name = ['disrupting', 'decreasing', 'no effect', 'increasing']
adjust_cof = len(group1) * (len(group1) - 1) / 2
for i in range(len(group1) - 1):
    for j in range(i + 1, len(group1)):
        odds, p = stats.fisher_exact([[group1[i], group2[i]], [group1[j], group2[j]]])
        print('{} and {}, OR: {:.3f}, p-value: {:.9f}'.format(group_name[i], group_name[j], odds, p * adjust_cof))
# stats.fisher_exact([[3015, 235], [916, 80]])
'''

'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
my_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
x, y = 'pred_class', 'sample_type'
df_1 = df_plus.groupby(y)[x].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()

plot = sns.barplot(x=x, y='percent', hue=y, data=df_1, order=my_order, palette=color_list3)
plot.set_ylim(0, 100)
# plot = sns.countplot(data=master_table, x='pred_most_common_type', ax=ax, order=my_order, hue='sample_type')
# plt.rcParams["axes.labelsize"] = 12

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

ax.set_xlabel('most common type of prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=18, fontweight='bold')

# statistical annotation
x1, x2 = 0, 2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = 70, 5, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x1 - 0.2, x1 + 0.2], [y, y], lw=1.5, c=col)
plt.plot([x2 - 0.2, x2 + 0.2], [y, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "OR = 1.30, P-value < 1e-9", ha='center', va='bottom', color=col)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_title('')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./cancer_via_att_plots_new/PPIunit_case_control_percent.tiff", dpi=png2.info['dpi'])
png1.close()
'''


master_table = pd.read_csv('cancer_mippi_done_via_att.tsv', sep='\t')
master_table['pred_most_common_type'] = master_table['pred_most_common']
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(0, 'disrupting')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(1, 'decreasing')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(2, 'no effect')
master_table['pred_most_common_type'] = master_table['pred_most_common_type'].replace(3, 'increasing')

'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
my_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
x, y = 'pred_most_common_type', 'sample_type'
df_1 = master_table.groupby(y)[x].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()

plot = sns.barplot(x=x, y='percent', hue=y, data=df_1, order=my_order, palette=color_list3)
plot.set_ylim(0, 100)
# plot = sns.countplot(data=master_table, x='pred_most_common_type', ax=ax, order=my_order, hue='sample_type')
# plt.rcParams["axes.labelsize"] = 12

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

ax.set_xlabel('most common type of prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# # statistical annotation
x1, x2 = 0, 2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = 70, 5, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x1 - 0.2, x1 + 0.2], [y, y], lw=1.5, c=col)
plt.plot([x2 - 0.2, x2 + 0.2], [y, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "OR = 1.309, P-value < 4.5e-8", ha='center', va='bottom', color=col)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_title('')

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./cancer_via_att_plots_new/pred_case_control_percent.tiff", dpi=png2.info['dpi'])
png1.close()
'''

'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']

x, y = 'SIFT_pred', 'pred_most_common_type'
df_1 = master_table[~(master_table['SIFT_pred'] == '.')]
df_1 = df_1.groupby(x)[y].value_counts(normalize=True)
df_1 = df_1.mul(100)
df_1 = df_1.rename('percent').reset_index()
test_plot = sns.barplot(data=df_1, x=x, y='percent', hue=y, ax=ax, hue_order=hue_order,palette=color_list1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for p in test_plot.patches:
    test_plot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
ax.legend().set_title('')
ax.legend(fontsize='15', bbox_to_anchor=(0.4, 0.85), loc=2, borderaxespad=0.)
# ax.legend(fontsize='15', loc='upper right')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set_xlabel('SIFT prediction', fontsize=18, fontweight='bold')
ax.set_ylabel('percent (%)', fontsize=18, fontweight='bold')
test_plot.set(xticklabels=['deleterious', ' tolerated'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

x1, x2 = -0.1, 0.9   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = 70, 3, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x1 - 0.2, x1 - 0.2, x1 + 0.2, x1 + 0.2], [y - h * 0.5, y, y, y - h * 0.5], lw=1.5, c=col)
plt.plot([x2 - 0.2, x2 - 0.2, x2 + 0.2, x2 + 0.2], [y - h * 0.5, y, y, y - h * 0.5], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "OR = 1.913, P-value < 1e-8", ha='center', va='bottom', color=col)

# plt.legend()

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./cancer_via_att_plots_new/pred_SIFT_percent.tiff", dpi=png2.info['dpi'])
png1.close()
'''

tmp_table = master_table[~(master_table['SIFT_score'] == '.')]
import joypy
'''
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.set(font_scale=2, style='white')
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
a = tmp_table[tmp_table['pred_most_common_type'] == 'disrupting']
b = tmp_table[tmp_table['pred_most_common_type'] == 'decreasing']
c = tmp_table[tmp_table['pred_most_common_type'] == 'no effect']
d = tmp_table[tmp_table['pred_most_common_type'] == 'increasing']
sns.kdeplot(a['SIFT_converted_rankscore'], label='disrupting', ax=ax)
sns.kdeplot(b['SIFT_converted_rankscore'], label='decreasing', ax=ax)
sns.kdeplot(c['SIFT_converted_rankscore'], label='no effect', ax=ax)
sns.kdeplot(d['SIFT_converted_rankscore'], label='increasing', ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
'''
hue_order = ['disrupting', 'decreasing', 'no effect', 'increasing']
a = tmp_table[tmp_table['pred_most_common_type'] == 'disrupting'].replace('disrupting', 0) 
b = tmp_table[tmp_table['pred_most_common_type'] == 'decreasing'].replace('decreasing', 1) 
c = tmp_table[tmp_table['pred_most_common_type'] == 'no effect'].replace('no effect', 2) 
d = tmp_table[tmp_table['pred_most_common_type'] == 'increasing'].replace('increasing', 3) 
new_df = pd.concat([a,b,c,d])
new_df['SIFT_converted_rankscore'] = new_df['SIFT_converted_rankscore'].astype(float)
print(new_df['pred_most_common_type'])
print(new_df['SIFT_converted_rankscore'])

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

png1 = io.BytesIO()
plt.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)
# Save as TIFF
png2.save("./cancer_via_att_plots_new/pred_SIFT_distribution_kde.tiff", dpi=png2.info['dpi'])
png1.close()