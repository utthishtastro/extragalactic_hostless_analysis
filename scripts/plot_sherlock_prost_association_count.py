import pandas as pd
import matplotlib.pyplot as plt


prost_results_file = "data/prost_cross_match.csv"
sherlock_results_file = "data/sherlock_cross_match.csv"
candidates_file = "data/candidates_tns_info.csv"


prost_df = pd.read_csv(prost_results_file, dtype=str)
sherlock_df = pd.read_csv(sherlock_results_file, dtype=str)
candidates_df = pd.read_csv(candidates_file, dtype=str)


category_mapping = {
    'dr9': 'DECaLS DR9',
    'dr2': 'Pan-STARRS DR2',
    'latest': 'GLADE+'
}
prost_df['best_cat_release'] = (
    prost_df['best_cat_release']
    .map(category_mapping)
    .fillna('Without host')
)

filtered_candidates = candidates_df[candidates_df['tns_classification_mapped'].notna()]

prost_merged = prost_df.merge(
    filtered_candidates,
    on='objectId',
    how='inner'
)

prost_before = prost_df['best_cat_release'].value_counts(dropna=False)
prost_after = prost_merged['best_cat_release'].value_counts(dropna=False)

prost_counts = pd.DataFrame({
    'Hostless candidates': prost_before,
    'Hostless with spectroscopic classification': prost_after
}).fillna(0)


sherlock_merged = sherlock_df.merge(
    filtered_candidates,
    on='objectId',
    how='inner'
)

def two_categories(counts):
    zero_count = counts.get('0', 0)
    rest_count = counts.sum() - zero_count
    return pd.Series({
        'Without association': zero_count,
        'With association': rest_count
    })

sherlock_before = two_categories(
    sherlock_df['catalogue_table_name'].value_counts(dropna=False)
)
sherlock_after = two_categories(
    sherlock_merged['catalogue_table_name'].value_counts(dropna=False)
)

sherlock_counts = pd.DataFrame({
    'Hostless candidates': sherlock_before,
    'Hostless with spectroscopic classification': sherlock_after
})


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 12), sharex=False
)


prost_counts.plot(kind='bar', ax=ax1)
ax1.set_title("Pröst host association counts")
ax1.set_ylabel("Count")
ax1.set_xlabel("")
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(title="")
ax1.tick_params(axis='x', labelrotation=0)


sherlock_counts.plot(kind='bar', ax=ax2)
ax2.set_title("Sherlock host association counts")
ax2.set_ylabel("Count")
ax2.set_xlabel("Category")
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(title="")
ax2.tick_params(axis='x', labelrotation=0)
plt.tight_layout()

plt.savefig("plots/host_association_count.pdf", bbox_inches="tight")
plt.savefig("plots/host_association_count.png", dpi=300, bbox_inches="tight")
plt.show()
