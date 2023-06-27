import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt


category_names = ['Deleterious', 'Tolerated']
results = {
    'disrupting': [38.6/(38.6+25.7), 25.7/(38.6+25.7)],
    'decreasing': [7.9/(7.9+6.0), 6.0/(7.9+6.0)],
    'no effect': [47.4/(47.4+60.4), 60.4/(47.4+60.4)],
    'increasing': [6.1/(6.1+7.9),7.9/(6.1+7.9)],
}
print(results)

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('tab20b')(
        np.linspace(0.15, 0.85, data.shape[1]))
    print(category_colors)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)

png1 = io.BytesIO()
plt.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("./cancer_via_att_plots_new/tcga.tiff", dpi=png2.info['dpi'])
png1.close()
