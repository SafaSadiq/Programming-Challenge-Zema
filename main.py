from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_PRICES_DIR = BASE_DIR / "historicalPriceData"
OUTPUT_DIR = BASE_DIR / "outputs"
REQUIRED_COLUMNS = {"Date", "SettlementPoint", "Price"}
HOUR_COLUMNS = [f"X{hour}" for hour in range(1, 25)]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_columns(dataframe: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def load_historical_prices(input_dir: Path | str = HISTORICAL_PRICES_DIR) -> pd.DataFrame:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Historical price directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Historical price path is not a directory: {input_path}")

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_path}")

    dataframes = []
    for csv_file in csv_files:
        dataframe = pd.read_csv(csv_file)
        validate_columns(dataframe)
        dataframe["source_file"] = csv_file.name
        dataframes.append(dataframe)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
    if combined_df["Date"].isna().any():
        raise ValueError("One or more Date values could not be parsed as datetimes.")
    if combined_df["Price"].isna().any():
        raise ValueError("One or more Price values are missing.")

    return combined_df

def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    columns: list[str] | None = None,
    *,
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    ensure_directory(output_dir)
    output_path = Path(output_dir) / filename
    if columns:
        df = df[columns]

    df.to_csv(output_path, index=False)

    return output_path

def compute_monthly_average_prices(combined_df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(combined_df)
    monthly_df = combined_df.copy()
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df["YearMonth"] = monthly_df["Date"].dt.to_period("M")

    monthly_avg = (
        monthly_df.groupby(["SettlementPoint", "YearMonth"], as_index=False)["Price"]
        .mean()
        .rename(columns={"Price": "AveragePrice"})
        .sort_values(["SettlementPoint", "YearMonth"])
        .reset_index(drop=True)
    )
    monthly_avg["Type"] = monthly_avg["SettlementPoint"].str.startswith("HB_").map({True: "Hub", False: "Load Zone"})
    monthly_avg["Year"] = monthly_avg["YearMonth"].dt.year
    monthly_avg["Month"] = monthly_avg["YearMonth"].dt.month
    return monthly_avg

def compute_hourly_volatility(combined_df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(combined_df)
    vol_df = combined_df.copy()
    vol_df["Date"] = pd.to_datetime(vol_df["Date"])
    vol_df = vol_df[vol_df["SettlementPoint"].str.startswith("HB_")]
    vol_df = vol_df[vol_df["Price"] > 0].sort_values(["SettlementPoint", "Date"])

    if vol_df.empty:
        raise ValueError("No positive hub prices were available to compute volatility.")

    vol_df["LogReturn"] = vol_df.groupby("SettlementPoint")["Price"].transform(lambda prices: np.log(prices).diff())
    vol_df["Year"] = vol_df["Date"].dt.year

    volatility = (
        vol_df.dropna(subset=["LogReturn"])
        .groupby(["SettlementPoint", "Year"])["LogReturn"]
        .std()
        .reset_index()
        .rename(columns={"LogReturn": "HourlyVolatility"})
        .sort_values(["SettlementPoint", "Year"])
        .reset_index(drop=True)
    )

    if volatility.empty:
        raise ValueError("Hourly volatility calculation produced no rows.")
    return volatility

def compute_max_volatility_by_year(volatility: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"SettlementPoint", "Year", "HourlyVolatility"}
    validate_columns(volatility, required_columns)
    return (
        volatility.loc[volatility.groupby("Year")["HourlyVolatility"].idxmax()]
        .sort_values("Year")
        .reset_index(drop=True)
    )


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def export_daily_price_files(combined_df: pd.DataFrame, output_dir: Path | str = OUTPUT_DIR) -> list[Path]:
    validate_columns(combined_df)
    output_folder = ensure_directory(Path(output_dir) / "formattedSpotHistory")

    price_df = combined_df.copy()
    price_df["DateTime"] = pd.to_datetime(price_df["Date"])
    price_df["Date"] = price_df["DateTime"].dt.strftime("%Y-%m-%d")
    price_df["HourColumn"] = "X" + (price_df["DateTime"].dt.hour + 1).astype(str)

    created_files = []
    for settlement_point, group in price_df.groupby("SettlementPoint"):
        daily_prices = (
            group.assign(Variable=settlement_point)
            .pivot_table(index=["Variable", "Date"], columns="HourColumn", values="Price", aggfunc="mean")
            .reindex(columns=HOUR_COLUMNS)
            .reset_index()
        )

        output_path = output_folder / f"spot_{safe_filename(settlement_point)}.csv"
        daily_prices.to_csv(output_path, index=False)
        created_files.append(output_path)

    return created_files


def monthly_average_price_plots(monthly_avg: pd.DataFrame, output_dir: Path | str = OUTPUT_DIR) -> dict[str, Path]:
    plot_output_dir = ensure_directory(Path(output_dir) / "BonusTask_MonthlyPrice")
    plot_df = monthly_avg.copy()
    plot_df["MonthStart"] = plot_df["YearMonth"].dt.to_timestamp()
    plot_df["SettlementType"] = plot_df["SettlementPoint"].str.startswith("HB_").map({True: "Hub", False: "Load Zone"})
    plot_df = plot_df.sort_values(["MonthStart", "SettlementPoint"])

    plt.style.use("seaborn-v0_8-whitegrid")
    color_palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors) + list(plt.get_cmap("Set1").colors)

    def plot_by_type(settlement_type: str, title: str, subtitle: str, filename: str) -> Path:
        subset = plot_df[plot_df["SettlementType"] == settlement_type]
        settlement_points = sorted(subset["SettlementPoint"].unique())
        if not settlement_points:
            raise ValueError(f"No settlement points found for type: {settlement_type}")

        fig, ax = plt.subplots(figsize=(15, 7.5))
        fig.patch.set_facecolor("#F5F7FA")
        ax.set_facecolor("#FFFFFF")

        for index, settlement_point in enumerate(settlement_points):
            series = subset[subset["SettlementPoint"] == settlement_point]
            ax.plot(
                series["MonthStart"],
                series["AveragePrice"],
                label=settlement_point,
                linewidth=1.0,
                color=color_palette[index % len(color_palette)],
                marker="o",
                markersize=4.8,
                markeredgewidth=0.7,
                markeredgecolor="white",
                alpha=0.95,
            )

        ax.set_title(title, fontsize=20, fontweight="bold", color="#111827", loc="left", pad=24)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=11, color="#475569", va="bottom")
        ax.set_xlabel("Month", fontsize=12, color="#334155", labelpad=12)
        ax.set_ylabel("Average Price ($/MWh)", fontsize=12, color="#334155", labelpad=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45, labelsize=10, colors="#334155")
        ax.tick_params(axis="y", labelsize=10, colors="#334155")
        ax.grid(True, axis="y", color="#CBD5E1", linewidth=0.9, alpha=0.65)
        ax.grid(True, axis="x", color="#E2E8F0", linewidth=0.6, alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.margins(x=0.015)

        legend = ax.legend(
            title="Settlement Point",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            borderpad=0.9,
            labelspacing=0.75,
            handlelength=2.6,
            fontsize=9,
            title_fontsize=10,
        )
        legend.get_frame().set_facecolor("#FFFFFF")
        legend.get_frame().set_edgecolor("#CBD5E1")
        legend.get_frame().set_linewidth(1.0)

        output_path = plot_output_dir / filename
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path

    return {
        "hub_plot": plot_by_type(
            "Hub",
            "Monthly Average Prices - Settlement Hubs",
            "Each curve is one HB_ settlement point, plotted chronologically from 2016 through 2019.",
            "SettlementHubAveragePriceByMonth.png",
        ),
        "load_zone_plot": plot_by_type(
            "Load Zone",
            "Monthly Average Prices - Load Zones",
            "Each curve is one LZ_ settlement point, plotted chronologically from 2016 through 2019.",
            "LoadZoneAveragePriceByMonth.png",
        ),
    }


def yearly_volatility_plots(volatility: pd.DataFrame, output_dir: Path | str = OUTPUT_DIR) -> dict[str, Path]:
    bonus_output_dir = ensure_directory(Path(output_dir) / "BonusTask_YearlyVolatility")
    volatility_plot_df = volatility.copy().sort_values(["Year", "SettlementPoint"])
    volatility_pivot = volatility_plot_df.pivot(index="SettlementPoint", columns="Year", values="HourlyVolatility").sort_index()
    volatility_by_year = volatility_pivot.T

    hub_order = [hub for hub in volatility_by_year.columns if hub != "HB_PAN"]
    if "HB_PAN" in volatility_by_year.columns:
        hub_order.append("HB_PAN")
    volatility_by_year = volatility_by_year[hub_order]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_facecolor("#FFFFFF")
    volatility_by_year.plot(kind="bar", ax=ax, width=0.82, colormap="tab20")
    ax.set_title("Hourly Price Volatility by Year and Settlement Hub", fontsize=18, fontweight="bold", color="#111827", loc="left", pad=18)
    ax.set_xlabel("Year", fontsize=12, color="#334155", labelpad=10)
    ax.set_ylabel("Hourly Volatility (Std. Dev. of Log Returns)", fontsize=12, color="#334155", labelpad=10)
    ax.tick_params(axis="x", rotation=0, labelsize=11, colors="#334155")
    ax.tick_params(axis="y", labelsize=10, colors="#334155")
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=0.9, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.legend(title="Settlement Hub", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9, title_fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    bar_chart_path = bonus_output_dir / "YearlyVolatilityBarChart.png"
    fig.savefig(bar_chart_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_facecolor("#FFFFFF")
    heatmap = ax.imshow(volatility_pivot.values, cmap="YlGnBu", aspect="auto", alpha=0.82)
    ax.set_title("Hourly Price Volatility Heatmap by Settlement Hub and Year", fontsize=18, fontweight="bold", color="#111827", loc="left", pad=18)
    ax.set_xlabel("Year", fontsize=12, color="#334155", labelpad=10)
    ax.set_ylabel("Settlement Hub", fontsize=12, color="#334155", labelpad=10)
    ax.set_xticks(np.arange(len(volatility_pivot.columns)))
    ax.set_xticklabels(volatility_pivot.columns, color="#334155")
    ax.set_yticks(np.arange(len(volatility_pivot.index)))
    ax.set_yticklabels(volatility_pivot.index, color="#334155")

    threshold = np.nanmean(volatility_pivot.values)
    for row_index in range(volatility_pivot.shape[0]):
        for col_index in range(volatility_pivot.shape[1]):
            value = volatility_pivot.iloc[row_index, col_index]
            text_color = "#0F172A" if value < threshold else "#FFFFFF"
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Hourly Volatility", color="#334155", labelpad=10)
    colorbar.ax.tick_params(colors="#334155")
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    heatmap_path = bonus_output_dir / "YearlyVolatilityHeatmap.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"bar_chart": bar_chart_path, "heatmap": heatmap_path}


def compute_hourly_shape_profiles(combined_df: pd.DataFrame, output_dir: Path | str = OUTPUT_DIR) -> tuple[list[Path], pd.DataFrame]:
    validate_columns(combined_df)
    profile_output_dir = ensure_directory(Path(output_dir) / "hourlyShapeProfiles")

    shape_df = combined_df.copy()
    shape_df["DateTime"] = pd.to_datetime(shape_df["Date"])
    shape_df["Month"] = shape_df["DateTime"].dt.month
    shape_df["DayOfWeek"] = shape_df["DateTime"].dt.day_name()
    shape_df["HourColumn"] = "X" + (shape_df["DateTime"].dt.hour + 1).astype(str)
    profile_index = pd.MultiIndex.from_product([range(1, 13), DAY_ORDER], names=["Month", "DayOfWeek"])

    profile_files = []
    profile_checks = []
    for settlement_point, group in shape_df.groupby("SettlementPoint"):
        fallback_hourly_average = group.groupby("HourColumn")["Price"].mean().reindex(HOUR_COLUMNS)
        if fallback_hourly_average.isna().any():
            raise ValueError(f"Overall hourly fallback profile is incomplete for {settlement_point}.")

        hourly_average = (
            group.pivot_table(index=["Month", "DayOfWeek"], columns="HourColumn", values="Price", aggfunc="mean")
            .reindex(profile_index)
            .reindex(columns=HOUR_COLUMNS)
        )

        fallback_profiles = pd.DataFrame(
            np.tile(fallback_hourly_average.values, (len(profile_index), 1)),
            index=profile_index,
            columns=HOUR_COLUMNS,
        )
        hourly_average = hourly_average.combine_first(fallback_profiles)
        hourly_average = hourly_average.T.fillna(hourly_average.mean(axis=1)).T

        profile_values = hourly_average.div(hourly_average.mean(axis=1), axis=0)
        profile_values["X24"] = 24 - profile_values[HOUR_COLUMNS[:-1]].sum(axis=1)

        profile = profile_values.reset_index()
        profile.insert(0, "SettlementPoint", settlement_point)
        output_path = profile_output_dir / f"profile_{settlement_point}.csv"
        profile.to_csv(output_path, index=False)
        profile_files.append(output_path)

        profile_checks.append(
            {
                "SettlementPoint": settlement_point,
                "Rows": len(profile),
                "MaxMeanDeviationFromOne": (profile[HOUR_COLUMNS].mean(axis=1) - 1).abs().max(),
                "MissingProfileRows": profile[HOUR_COLUMNS].isna().any(axis=1).sum(),
            }
        )

    profile_check_df = pd.DataFrame(profile_checks)
    if len(profile_files) != shape_df["SettlementPoint"].nunique():
        raise RuntimeError("Hourly shape profile export did not create one file per settlement point.")
    if not (profile_check_df["Rows"] == 84).all():
        raise RuntimeError("One or more hourly shape profile files does not contain 84 profiles.")
    if profile_check_df["MissingProfileRows"].sum() != 0:
        raise RuntimeError("One or more hourly shape profiles contains missing values.")

    return profile_files, profile_check_df


def check_loaded_data(combined_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_columns(combined_df)
    check_df = combined_df.copy()
    check_df["Date"] = pd.to_datetime(check_df["Date"])

    summary = pd.DataFrame(
        [
            {"Check": "Rows", "Value": len(check_df)},
            {"Check": "Settlement points", "Value": check_df["SettlementPoint"].nunique()},
            {"Check": "First timestamp", "Value": check_df["Date"].min()},
            {"Check": "Last timestamp", "Value": check_df["Date"].max()},
            {"Check": "Duplicate Date/SettlementPoint rows", "Value": check_df.duplicated(["Date", "SettlementPoint"]).sum()},
            {"Check": "Missing prices", "Value": check_df["Price"].isna().sum()},
            {"Check": "Zero or negative prices", "Value": (check_df["Price"] <= 0).sum()},
        ]
    )

    missing_records = []
    for settlement_point, group in check_df.groupby("SettlementPoint"):
        settlement_timestamps = pd.DatetimeIndex(group["Date"].sort_values().unique())
        expected_timestamps = pd.date_range(settlement_timestamps.min(), settlement_timestamps.max(), freq="h")
        missing_timestamps = expected_timestamps.difference(settlement_timestamps)
        for missing_timestamp in missing_timestamps:
            missing_records.append(
                {
                    "SettlementPoint": settlement_point,
                    "MissingTimestamp": missing_timestamp,
                    "Month": missing_timestamp.month,
                    "Hour": missing_timestamp.hour,
                }
            )

    missing_hours = pd.DataFrame(missing_records)
    missing_summary = (
        missing_hours.groupby("SettlementPoint")
        .size()
        .reset_index(name="MissingHours")
        .sort_values(["MissingHours", "SettlementPoint"], ascending=[False, True])
        if not missing_hours.empty
        else pd.DataFrame(columns=["SettlementPoint", "MissingHours"])
    )

    return {"summary": summary, "missing_hours": missing_hours, "missing_summary": missing_summary}


# def save_open_ended_analysis_plots(
#     combined_df: pd.DataFrame,
#     monthly_avg: pd.DataFrame,
#     output_dir: Path | str = OUTPUT_DIR,
# ) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
#     validate_columns(combined_df)
#     analysis_output_dir = ensure_directory(Path(output_dir) / "OpenEndedAnalysis")
#     data_checks = check_loaded_data(combined_df)

#     monthly_plot_df = monthly_avg.copy()
#     monthly_plot_df["MonthStart"] = monthly_plot_df["YearMonth"].dt.to_timestamp()
#     monthly_plot_df["SettlementType"] = monthly_plot_df["SettlementPoint"].str.startswith("HB_").map({True: "Hub", False: "Load Zone"})
#     hub_load_zone_mapping = {
#         "HB_HOUSTON": "LZ_HOUSTON",
#         "HB_NORTH": "LZ_NORTH",
#         "HB_SOUTH": "LZ_SOUTH",
#         "HB_WEST": "LZ_WEST",
#     }

#     monthly_price_pivot = monthly_plot_df.pivot(index="MonthStart", columns="SettlementPoint", values="AveragePrice")
#     missing_mapping_points = sorted(
#         settlement_point
#         for pair in hub_load_zone_mapping.items()
#         for settlement_point in pair
#         if settlement_point not in monthly_price_pivot.columns
#     )
#     if missing_mapping_points:
#         raise ValueError(f"Missing settlement points needed for hub/load-zone mapping: {missing_mapping_points}")

#     basis_records = []
#     for hub, load_zone in hub_load_zone_mapping.items():
#         pair_basis = pd.DataFrame(
#             {
#                 "MonthStart": monthly_price_pivot.index,
#                 "SettlementHub": hub,
#                 "LoadZone": load_zone,
#                 "HubPrice": monthly_price_pivot[hub].values,
#                 "LoadZonePrice": monthly_price_pivot[load_zone].values,
#             }
#         )
#         pair_basis["Pair"] = f"{hub} vs {load_zone}"
#         pair_basis["LoadZoneMinusHubBasis"] = pair_basis["LoadZonePrice"] - pair_basis["HubPrice"]
#         pair_basis["AbsBasis"] = pair_basis["LoadZoneMinusHubBasis"].abs()
#         basis_records.append(pair_basis)

#     basis = pd.concat(basis_records, ignore_index=True)
#     congestion_threshold = basis["AbsBasis"].mean() + 2 * basis["AbsBasis"].std()
#     congestion_periods = basis[basis["AbsBasis"] >= congestion_threshold].sort_values("AbsBasis", ascending=False)

#     volatility_event_df = combined_df.copy()
#     volatility_event_df["Date"] = pd.to_datetime(volatility_event_df["Date"])
#     volatility_event_df = volatility_event_df.sort_values(["SettlementPoint", "Date"])
#     volatility_event_df["PriceChange"] = volatility_event_df.groupby("SettlementPoint")["Price"].diff()
#     volatility_event_df["AbsPriceChange"] = volatility_event_df["PriceChange"].abs()
#     volatility_event_df["PricePeriod"] = np.where(volatility_event_df["Price"] <= 0, "Negative price hours", "Positive price hours")
#     volatility_during_negative_prices = (
#         volatility_event_df.dropna(subset=["AbsPriceChange"])
#         .groupby(["SettlementPoint", "PricePeriod"])
#         .agg(
#             Observations=("AbsPriceChange", "size"),
#             AverageAbsHourlyPriceChange=("AbsPriceChange", "mean"),
#             StdHourlyPriceChange=("PriceChange", "std"),
#             MinPrice=("Price", "min"),
#         )
#         .reset_index()
#     )
#     negative_price_summary = (
#         volatility_during_negative_prices.pivot(
#             index="SettlementPoint",
#             columns="PricePeriod",
#             values="AverageAbsHourlyPriceChange",
#         )
#         .reset_index()
#         .rename_axis(None, axis=1)
#     )
#     negative_price_summary = negative_price_summary.dropna(subset=["Negative price hours"])
#     negative_price_summary["NegativeVsPositiveChangeRatio"] = (
#         negative_price_summary["Negative price hours"] / negative_price_summary["Positive price hours"]
#     )
#     negative_hour_counts = (
#         volatility_event_df[volatility_event_df["Price"] <= 0]
#         .groupby("SettlementPoint")
#         .size()
#         .reset_index(name="NegativeHourCount")
#     )
#     negative_price_summary = (
#         negative_price_summary.merge(negative_hour_counts, on="SettlementPoint", how="left")
#         .sort_values("NegativeVsPositiveChangeRatio", ascending=False)
#         .reset_index(drop=True)
#     )
#     negative_price_summary = negative_price_summary[
#         negative_price_summary["SettlementPoint"].str.startswith("HB_")
#     ].reset_index(drop=True)

#     for obsolete_filename in [
#         "MissingHoursBySettlementPoint.png",
#         "AnnotatedMonthlyAveragePrice_August2019Heatwave.png",
#         "AverageHubVsLoadZonePrices.png",
#     ]:
#         obsolete_path = analysis_output_dir / obsolete_filename
#         if obsolete_path.exists():
#             obsolete_path.unlink()

#     plt.style.use("seaborn-v0_8-whitegrid")
#     plot_paths: dict[str, Path] = {}

#     fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
#     fig.patch.set_facecolor("#F8FAFC")
#     for ax, (hub, load_zone) in zip(axes.flat, hub_load_zone_mapping.items()):
#         ax.set_facecolor("#FFFFFF")
#         ax.plot(monthly_price_pivot.index, monthly_price_pivot[hub], label=hub, color="#2563EB", linewidth=2.2, marker="o", markersize=3.8)
#         ax.plot(monthly_price_pivot.index, monthly_price_pivot[load_zone], label=load_zone, color="#F97316", linewidth=2.2, marker="s", markersize=3.8)
#         ax.set_title(f"{hub} vs {load_zone}", fontsize=13, fontweight="bold", color="#111827", loc="left", pad=10)
#         ax.set_ylabel("Average Price ($/MWh)", color="#334155")
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#         ax.tick_params(axis="x", rotation=35, labelsize=9, colors="#334155")
#         ax.tick_params(axis="y", labelsize=9, colors="#334155")
#         ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8)
#         ax.grid(True, color="#CBD5E1", alpha=0.55)
#         ax.spines[["top", "right"]].set_visible(False)
#     fig.suptitle("Mapped Hub vs Load-Zone Monthly Average Prices", fontsize=18, fontweight="bold", color="#111827", x=0.02, ha="left")
#     plt.tight_layout()
#     plot_paths["hub_vs_load_zone"] = analysis_output_dir / "MappedHubLoadZoneAveragePrices.png"
#     fig.savefig(plot_paths["hub_vs_load_zone"], dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close(fig)

#     basis_heatmap = basis.pivot(index="Pair", columns="MonthStart", values="LoadZoneMinusHubBasis")
#     fig, ax = plt.subplots(figsize=(15, 5.6))
#     fig.patch.set_facecolor("#F8FAFC")
#     ax.set_facecolor("#FFFFFF")
#     max_abs_basis = basis["AbsBasis"].max()
#     heatmap = ax.imshow(basis_heatmap.values, aspect="auto", cmap="RdBu_r", vmin=-max_abs_basis, vmax=max_abs_basis, alpha=0.86)
#     ax.set_title("Mapped Hub-to-Load-Zone Basis and High-Congestion Periods", fontsize=17, fontweight="bold", color="#111827", loc="left", pad=16)
#     ax.set_xlabel("Month", color="#334155", labelpad=10)
#     ax.set_ylabel("Mapped pair", color="#334155", labelpad=10)
#     ax.set_xticks(np.arange(len(basis_heatmap.columns))[::3])
#     ax.set_xticklabels([timestamp.strftime("%Y-%m") for timestamp in basis_heatmap.columns[::3]], rotation=45, ha="right", color="#334155")
#     ax.set_yticks(np.arange(len(basis_heatmap.index)))
#     ax.set_yticklabels(basis_heatmap.index, color="#334155")
#     for _, row in congestion_periods.iterrows():
#         row_index = basis_heatmap.index.get_loc(row["Pair"])
#         col_index = basis_heatmap.columns.get_loc(row["MonthStart"])
#         ax.scatter(col_index, row_index, marker="o", s=90, facecolors="none", edgecolors="#111827", linewidths=1.8)
#     colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02)
#     colorbar.set_label("Load Zone - Hub ($/MWh)", color="#334155", labelpad=10)
#     colorbar.ax.tick_params(colors="#334155")
#     ax.spines[:].set_visible(False)
#     plt.tight_layout()
#     plot_paths["basis_congestion"] = analysis_output_dir / "HubLoadZoneBasisCongestion.png"
#     fig.savefig(plot_paths["basis_congestion"], dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close(fig)

#     fig, ax = plt.subplots(figsize=(12, 6.5))
#     fig.patch.set_facecolor("#F8FAFC")
#     ax.set_facecolor("#FFFFFF")
#     bar_positions = np.arange(len(negative_price_summary))
#     ax.bar(
#         bar_positions,
#         negative_price_summary["NegativeVsPositiveChangeRatio"],
#         color="#A78BFA",
#         edgecolor="#312E81",
#         linewidth=0.8,
#         label="Negative/positive volatility ratio",
#     )
#     ax.axhline(1, color="#0F172A", linewidth=1.1, linestyle="--", label="Same as positive-price hours")
#     count_ax = ax.twinx()
#     count_ax.plot(
#         bar_positions,
#         negative_price_summary["NegativeHourCount"],
#         color="#F97316",
#         marker="o",
#         linewidth=2.2,
#         label="Negative price hours",
#     )
#     for position, count in zip(bar_positions, negative_price_summary["NegativeHourCount"]):
#         count_ax.annotate(
#             f"{int(count)}h",
#             xy=(position, count),
#             xytext=(0, 7),
#             textcoords="offset points",
#             ha="center",
#             fontsize=9,
#             color="#9A3412",
#         )
#     ax.set_title("Volatility During Negative Price Periods", fontsize=17, fontweight="bold", color="#111827", loc="left", pad=16)
#     ax.set_xlabel("Settlement Hub", color="#334155", labelpad=10)
#     ax.set_ylabel("Avg abs hourly price change ratio", color="#334155", labelpad=10)
#     count_ax.set_ylabel("Negative price hours", color="#9A3412", labelpad=10)
#     ax.set_xticks(bar_positions)
#     ax.set_xticklabels(negative_price_summary["SettlementPoint"], rotation=35, ha="right")
#     ax.tick_params(axis="x", labelsize=9, colors="#334155")
#     ax.tick_params(axis="y", colors="#334155")
#     count_ax.tick_params(axis="y", colors="#9A3412")
#     ax_handles, ax_labels = ax.get_legend_handles_labels()
#     count_handles, count_labels = count_ax.get_legend_handles_labels()
#     ax.legend(ax_handles + count_handles, ax_labels + count_labels, frameon=True, facecolor="white", edgecolor="#CBD5E1")
#     ax.grid(True, axis="y", color="#CBD5E1", alpha=0.65)
#     ax.spines[["top", "right"]].set_visible(False)
#     count_ax.spines[["top"]].set_visible(False)
#     plt.tight_layout()
#     plot_paths["negative_price_volatility"] = analysis_output_dir / "VolatilityDuringNegativePricePeriods.png"
#     fig.savefig(plot_paths["negative_price_volatility"], dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close(fig)

#     tables = {
#         "data_check_summary": data_checks["summary"],
#         "missing_hours": data_checks["missing_hours"],
#         "missing_summary": data_checks["missing_summary"],
#         "hub_load_zone_mapping": pd.DataFrame(
#             [{"SettlementHub": hub, "LoadZone": load_zone} for hub, load_zone in hub_load_zone_mapping.items()]
#         ),
#         "basis": basis,
#         "congestion_periods": congestion_periods,
#         "volatility_during_negative_prices": volatility_during_negative_prices,
#         "negative_price_volatility_summary": negative_price_summary,
#     }
#     return plot_paths, tables



def run_all(input_dir: Path | str = HISTORICAL_PRICES_DIR, output_dir: Path | str = OUTPUT_DIR) -> dict[str, object]:
    output_path = ensure_directory(output_dir)
    combined_df = load_historical_prices(input_dir)
    monthly_avg = compute_monthly_average_prices(combined_df)
    volatility = compute_hourly_volatility(combined_df)
    max_vol_by_year = compute_max_volatility_by_year(volatility)
    daily_price_files = export_daily_price_files(combined_df, output_path)
    monthly_plot_paths = save_dataframe(
        monthly_avg,
        "AveragePriceByMonth",
        ["SettlementPoint", "Year", "Month", "AveragePrice"],
        output_dir=output_path,
    )
    yearly_volatility_plot_paths = save_dataframe(volatility, "HourlyVolatilityByYear", output_dir=output_path)
    profile_files, profile_check_df = compute_hourly_shape_profiles(combined_df, output_path)
    # open_ended_plot_paths, open_ended_tables = save_open_ended_analysis_plots(combined_df, monthly_avg, output_path)

    outputs = {
        "combined_df": combined_df,
        "monthly_avg": monthly_avg,
        "volatility": volatility,
        "max_vol_by_year": max_vol_by_year,
        "average_price_csv": save_dataframe(
            monthly_avg,
            "AveragePriceByMonth",
            ["SettlementPoint", "Year", "Month", "AveragePrice"],
            output_dir=output_path,
        ),
        "hourly_volatility_csv": save_dataframe(volatility, "HourlyVolatilityByYear", output_dir=output_path),
        "max_volatility_csv": save_dataframe(max_vol_by_year, "MaxVolatilityByYear", output_dir=output_path),
        "daily_price_files": daily_price_files,
        "monthly_plot_paths": monthly_plot_paths,
        "yearly_volatility_plot_paths": yearly_volatility_plot_paths,
        "profile_files": profile_files,
        "profile_check_df": profile_check_df,
        # "open_ended_plot_paths": open_ended_plot_paths,
        # "open_ended_tables": open_ended_tables,
    }
    return outputs


if __name__ == "__main__":
    results = run_all()
    print("Completed all tasks.")
    print(f"Rows loaded: {len(results['combined_df'])}")
    print(f"Settlement points: {results['combined_df']['SettlementPoint'].nunique()}")
    print(f"Output directory: {OUTPUT_DIR}")