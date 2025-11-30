import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

df = pd.read_csv("data/candidates_tns_info.csv")

plt.figure(figsize=(12, 7))
df['tns_classification_mapped'].value_counts().sort_values().plot(
    kind='barh'
)

plt.title("Distribution of transient classes")
plt.xlabel("Count")
plt.ylabel("Transient class")
plt.tight_layout()

plt.savefig("plots/class_distribution.pdf", bbox_inches="tight")
plt.savefig("plots/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


df_plot = df.dropna(subset=['tns_classification_mapped', 'tns_redshift'])

N = 1
counts = df_plot['tns_classification_mapped'].value_counts()
classes_to_keep = counts[counts >= N].index
df_plot = df_plot[df_plot['tns_classification_mapped'].isin(classes_to_keep)]

plt.figure(figsize=(12, 7))
sns.stripplot(
    x='tns_redshift',
    y='tns_classification_mapped',
    data=df_plot,
    jitter=False,
    alpha=1,
    order=df_plot.groupby('tns_classification_mapped')['tns_redshift']
                .median()
                .sort_values()
                .index
)

plt.title("Redshift vs transient class")
plt.xlabel("Redshift")
plt.ylabel("Transient class")
plt.tight_layout()

plt.savefig("plots/redshift_vs_class.pdf", bbox_inches="tight")
plt.savefig("plots/redshift_vs_class.png", dpi=300, bbox_inches="tight")
plt.show()
