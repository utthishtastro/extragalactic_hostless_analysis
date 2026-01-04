import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

df = pd.read_csv("data/candidates_tns_info.csv")

df_plot = df.dropna(subset=['tns_classification_mapped'])

# Keep classes with at least N counts
N = 1
counts = df_plot['tns_classification_mapped'].value_counts()
classes_to_keep = counts[counts >= N].index
df_plot = df_plot[df_plot['tns_classification_mapped'].isin(classes_to_keep)]

class_order = (
    df_plot.groupby('tns_classification_mapped')['tns_redshift']
    .median()
    .sort_values()
    .index
)


counts_ordered = (
    df_plot['tns_classification_mapped']
    .value_counts()
    .reindex(class_order)
)

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 7),
    sharey=True,
    gridspec_kw={'width_ratios': [1, 2]}
)


ax1.barh(class_order, counts_ordered)
ax1.set_title("Class distribution")
ax1.set_xlabel("Count")
ax1.set_ylabel("Transient class")
ax1.margins(y=0)


sns.stripplot(
    x='tns_redshift',
    y='tns_classification_mapped',
    data=df_plot,
    order=class_order,
    jitter=False,
    alpha=1,
    ax=ax2
)

ax2.set_title("Redshift vs class")
ax2.set_xlabel("Redshift")
ax2.set_ylabel("")
ax2.tick_params(labelleft=False)
ax2.margins(y=0)

plt.savefig("plots/class_distribution_and_redshift.pdf", bbox_inches="tight")
plt.savefig("plots/class_distribution_and_redshift.png", dpi=300, bbox_inches="tight")
plt.show()
