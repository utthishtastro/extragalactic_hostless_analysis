import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sherlock_results_file = "data/sherlock_cross_match.csv"
prost_results_file = "data/prost_cross_match.csv"

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

sherlock_columns = {
    "catalogue": "catalogue_table_name",
    "host_column": "catalogue_object_id",
    "PS1": "PS1",
    "dr9": "none"
}

prost_columns = {
    "catalogue": "best_cat_release",
    "host_column": "host_objID",
    "PS1": "dr2",
    "dr9": "dr9"
}

host_meta_data_mapping = {
    "dr9": {
        "file_name": "data/host_association/legacy_dr9.csv",
        "host_column": "ls_id",
        "mag_column": "mag_r"
    },
    "PS1": {
        "file_name": "data/host_association/panstarrs_dr2.csv",
        "host_column": "objID",
        "mag_column": "rMeanKronMag"
    }
}

survey_name_map = {
    "PS1": "Pan-STARRS DR2",
    "dr9": "Legacy Survey DECaLS DR9"
}

# ZTF r-band limiting mag
ztf_r_limit = 20.7


def load_results(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, dtype=str)


def load_host_metadata(mapping: dict) -> dict:
    host_dfs = {}

    for survey, meta in mapping.items():
        df = pd.read_csv(meta["file_name"], dtype=str)
        df = df[[meta["host_column"], meta["mag_column"]]].copy()
        df[meta["mag_column"]] = pd.to_numeric(df[meta["mag_column"]], errors="coerce")

        if survey == "PS1":
            df = (
                df.sort_values(meta["mag_column"])
                  .drop_duplicates(subset=meta["host_column"], keep="first")
            )

        host_dfs[survey] = df

    return host_dfs


def merge_with_host_metadata(
    results_df: pd.DataFrame,
    columns_mapping: dict,
    host_dfs: dict,
    host_meta_mapping: dict
) -> pd.DataFrame:

    merged_list = []

    for survey, cat_val in columns_mapping.items():
        if survey in ["catalogue", "host_column"]:
            continue
        if cat_val.lower() == "none":
            continue

        df_subset = results_df[
            results_df[columns_mapping["catalogue"]] == cat_val
        ].copy()

        hosts = host_dfs[survey]
        meta = host_meta_mapping[survey]

        merged = df_subset.merge(
            hosts,
            left_on=columns_mapping["host_column"],
            right_on=meta["host_column"],
            how="left"
        )

        merged["survey"] = survey
        merged_list.append(merged)

    return pd.concat(merged_list, ignore_index=True) if merged_list else pd.DataFrame()


def standardize_magnitude(df: pd.DataFrame, host_meta_mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["magnitude"] = df.apply(
        lambda row: row[host_meta_mapping[row["survey"]]["mag_column"]],
        axis=1
    )
    return df


def filter_magnitude(df: pd.DataFrame, min_val=10, max_val=30) -> pd.DataFrame:
    return df[(df["magnitude"] >= min_val) & (df["magnitude"] <= max_val)]


def percentage_outside_ztf_limit(df: pd.DataFrame, ztf_limit: float) -> pd.DataFrame:
    results = []

    for survey in df["survey"].unique():
        survey_df = df[df["survey"] == survey]

        total = len(survey_df)
        if total == 0:
            continue

        outside = (survey_df["magnitude"] > ztf_limit).sum()
        percentage = 100.0 * outside / total

        results.append({
            "survey": survey_name_map.get(survey, survey),
            "total_hosts": total,
            "outside_ztf_limit": outside,
            "percentage_outside": percentage
        })

    return pd.DataFrame(results)


def plot_magnitude_distribution(
    df: pd.DataFrame,
    ztf_limit: float = None,
    show: bool = True
):
    surveys = list(df["survey"].unique())

    fig, axes = plt.subplots(
        nrows=len(surveys),
        ncols=1,
        figsize=(8, 5 * len(surveys)),
        sharex=True
    )

    if len(surveys) == 1:
        axes = [axes]

    for ax, survey in zip(axes, surveys):

        survey_data = df[df["survey"] == survey]
        mag_col = host_meta_data_mapping[survey]["mag_column"]
        survey_name = survey_name_map.get(survey, survey)

        sns.histplot(
            data=survey_data,
            x="magnitude",
            bins=30,
            color="tab:blue",
            alpha=0.8,
            ax=ax
        )

        if ztf_limit is not None:
            ax.axvline(
                ztf_limit,
                color="red",
                linestyle="--",
                label=f"ZTF r-limit ~ {ztf_limit}"
            )
            ax.legend()

        ax.set_ylabel("Number of Hosts")
        ax.set_title(f"Host Magnitude Distribution — {survey_name}")

    axes[-1].set_xlabel("Host magnitude")

    plt.tight_layout()

    png_path = os.path.join(plots_dir, "host_magnitude_distribution.png")
    pdf_path = os.path.join(plots_dir, "host_magnitude_distribution.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


sherlock_df = load_results(sherlock_results_file)
prost_df = load_results(prost_results_file)

host_dfs = load_host_metadata(host_meta_data_mapping)

sherlock_merged = merge_with_host_metadata(
    sherlock_df, sherlock_columns, host_dfs, host_meta_data_mapping
)
prost_merged = merge_with_host_metadata(
    prost_df, prost_columns, host_dfs, host_meta_data_mapping
)

all_merged = pd.concat([sherlock_merged, prost_merged], ignore_index=True)

all_merged = standardize_magnitude(all_merged, host_meta_data_mapping)
all_merged_filtered = filter_magnitude(all_merged, 10, 30)

plot_magnitude_distribution(
    all_merged_filtered,
    ztf_limit=ztf_r_limit,
    show=True
)

ztf_stats = percentage_outside_ztf_limit(
    all_merged_filtered,
    ztf_limit=ztf_r_limit
)

print("Percentage of hosts fainter than the ZTF limit:")
print(ztf_stats.to_string(index=False, float_format="%.2f"))
