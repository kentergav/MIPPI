import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
#sns.set(style="darkgrid")
#sns.set(style="whitegrid")
#sns.set_style("white")
sns.set(style="whitegrid",font_scale=2)

# df_path = r'../../../data/raw/raw_s51_0805_p.csv'
df_path = r'skempi2_via_att_plots_new/ddg_dataset.pickle'
df = pd.read_pickle(df_path)
df.reset_index()
print(df.shape)

plt.figure(figsize=(10, 8), dpi=300)
pred_list = ['disrupting', 'decreasing', 'no effect', 'increasing']
#color_list = ['#8F6798', '#83A4BB', '#86D4AF', '#FFF17C']
color_list = ['#FFF17C', '#86D4AF', '#83A4BB', '#8F6798']
sns.boxplot(y="cv5_class", x="ddg", data=df, showfliers = False, palette=color_list, orient="h")
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
#sns.boxplot(y="cv5_class", x="ddg", data=df, showfliers = False, palette=color_list, notch=True, orient="h")

#moving the rain below the boxplot
dx = "ddg"; dy = "cv5_class"; ort = "h"; pal = color_list; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))

ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort, move = .2)

#plt.title("Figure P8\n Rainclouds with Shifted Rain")
if savefigs:
    plt.savefig('./skempi2_via_att_plots_new/raincloud.png', bbox_inches='tight')
    '''