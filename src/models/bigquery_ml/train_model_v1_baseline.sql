/* ==========================================================================
   MODEL TRAINING: BASELINE BENCHMARK COMPARISON
   Version: V1 (Paper Replication & Linear vs. Log-Linear Test)

   Models:
     1. MODEL A: Linear Regression (Target: Raw Price)
     2. MODEL B: Log-Linear Regression (Target: LN(Price))

   Objective:
     1. Replicate the methodology of the reference study (Guanajuato) using
        similar variables and a consolidated crime metric.
     2. Demonstrate the statistical superiority of the Log-Linear approach
        for skewed real estate price distributions.
     3. Establish a baseline R2 score to beat with advanced ML (XGBoost) later.

   Feature Engineering Notes:
     - Crime Variable: Aggregated (Burglary + Violence + Vehicle + Passersby)
       to mimic the reference paper's likely methodology.
     - Exclusions: Business theft and Public Transport theft (irrelevant for housing).
     - Data Source: view_training_data_cleaned (Outliers removed).
   ========================================================================== */

CREATE OR REPLACE MODEL `real-estate-qro.queretaro_data_marts.model_baseline_linear`
OPTIONS(
  -- MODEL TYPE: Standard OLS Linear Regression
  model_type = 'LINEAR_REG',

  -- TARGET: We predict the raw price directly
  input_label_cols = ['target_price'],

  -- DATA SPLIT: BigQuery automatically handles Train/Test split
  -- It reserves ~10-20% of data to evaluate quality later.
  data_split_method = 'AUTO_SPLIT'
) AS

SELECT
  target_price,
  feat_bedrooms,
  feat_bathrooms,
  feat_parking_spots,
  feat_m2_constructed,
  feat_has_garden,
  feat_dist_center,
  feat_dist_supermarket,
  feat_dist_park,
  feat_crime_homicide,
  feat_crime_consolidated
FROM
  `real-estate-qro.queretaro_data_marts.view_training_data_cleaned`;

CREATE OR REPLACE MODEL `real-estate-qro.queretaro_data_marts.model_baseline_log_linear`
OPTIONS(
  -- MODEL TYPE: Log-Linear Regression
  model_type = 'LINEAR_REG',

  -- TARGET: We predict LN(Price)
  input_label_cols = ['ln_price'],

  data_split_method = 'AUTO_SPLIT'
) AS

SELECT
  LN(target_price) AS ln_price,
  feat_bedrooms,
  feat_bathrooms,
  feat_parking_spots,
  feat_m2_constructed,
  feat_has_garden,
  feat_dist_center,
  feat_dist_supermarket,
  feat_dist_park,
  feat_crime_homicide,
  feat_crime_consolidated
FROM
  `real-estate-qro.queretaro_data_marts.view_training_data_cleaned`;