"""
data/features/temporal_features.py — Step 6 (temporal group): known-at-inference calendar features.

Join gold.calendar_date for DOW, week-of-year, month, quarter, fiscal period,
day-of-month, payday flag, weekend flag, season. Known-at-inference for the
full 140-day horizon (feature_engineering_spec.md Step 6).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Known-at-inference: available for the full 140-day horizon (feature_engineering_spec.md §6).
# config/seed_defaults.py imports this exact list when seeding config_feature.
KNOWN_AT_INFERENCE_COLUMNS: list[str] = [
    "dow", "week_of_year", "month", "quarter", "fiscal_period",
    "day_of_month", "payday_flag", "weekend_flag", "season",
]


def add_temporal_features(spark: "SparkSession", catalog: str, df: "DataFrame") -> "DataFrame":
    """Join `{catalog}.gold.calendar_date` on date_id and attach the calendar/temporal columns.

    Verified column mapping against your_catalog.gold.calendar_date (2025-07-02):
      gold.calendar_date.weekday      -> dow          (source: 1=Sun…7=Sat; converted to ISO: 1=Mon…7=Sun)
      gold.calendar_date.week         -> week_of_year
      gold.calendar_date.day          -> day_of_month
      gold.calendar_date.month        -> month        (exists as-is)
      gold.calendar_date.quarter      -> quarter      (exists as-is)
    Derived:
      weekend_flag  -- CASE on weekday_name IN ('Saturday','Sunday')
      season        -- meteorological: DJF=winter, MAM=spring, JJA=summer, SON=autumn
    Not in gold.calendar_date (NULL placeholders — update if a source becomes available):
      fiscal_period -- retailer-specific fiscal calendar (varies by retailer); not present
                        in gold.calendar_date — populate from your own fiscal
                        calendar table if needed
      payday_flag   -- derived: day_of_month >= 25 OR <= 5, excluding weekends
                          (generic end-of-month / start-of-month pay cycle heuristic)
    """
    df.createOrReplaceTempView("_temporal_features_input")
    return spark.sql(f"""
        SELECT t.*,
               CASE WHEN d.weekday = 1 THEN 7 ELSE d.weekday - 1 END          AS dow,
               d.week                                                         AS week_of_year,
               d.month,
               d.quarter,
               d.day                                                          AS day_of_month,
               CASE WHEN LOWER(d.weekday_name) IN ('saturday', 'sunday')
                    THEN 1 ELSE 0 END                                         AS weekend_flag,
               CASE
                   WHEN d.month IN (12, 1, 2) THEN 'winter'
                   WHEN d.month IN (3, 4, 5)  THEN 'spring'
                   WHEN d.month IN (6, 7, 8)  THEN 'summer'
                   ELSE 'autumn'
               END                                                            AS season,
               CAST(NULL AS INT)                                              AS fiscal_period,
               CASE WHEN (d.day >= 25 OR d.day <= 5)
                         AND LOWER(d.weekday_name) NOT IN ('saturday', 'sunday')
                    THEN 1 ELSE 0 END                                         AS payday_flag
        FROM _temporal_features_input t
        JOIN {catalog}.gold.calendar_date d ON t.date_id = d.date_id
    """)
