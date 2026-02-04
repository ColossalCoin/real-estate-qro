/* ==========================================================================
   FINAL STEP: CLEANING OUTLIERS (TRAINING SET PREPARATION)
   Objective: Filter statistical garbage to prevent model confusion.
   Note: Log transformation is NOT applied here to allow for "Raw" scenario testing.
   ========================================================================== */

CREATE OR REPLACE VIEW `real-estate-qro.queretaro_data_marts.view_training_data_cleaned` AS

SELECT
  listing_id,

  -- TARGET VARIABLE (Raw Price)
  -- We keep the original price distribution for the baseline model.
  target_price,

  -- PHYSICAL FEATURES
  feat_m2_constructed,
  feat_m2_terrain,
  feat_bedrooms,
  feat_bathrooms,
  feat_parking_spots,
  feat_is_new,

  -- AMENITIES
  feat_has_security,
  feat_has_garden,
  feat_has_pool,
  feat_has_gym,
  feat_has_kitchen,
  feat_has_terrace,

  -- LOCATION
  feat_municipality,
  feat_neighborhood,

  -- CRIME CONTEXT
  feat_crime_total,
  feat_crime_residential

FROM `real-estate-qro.queretaro_data_marts.obt_listings_valuation_features`

WHERE
  -- 1. PRICE FILTER (Reasonable range for housing market)
  -- Removing outliers < 300k (likely errors/land) and > 80M (extreme luxury/commercial)
  target_price BETWEEN 300000 AND 80000000

  -- 2. SURFACE FILTER (Removing the INT32 Max bug and 1m2 errors)
  AND feat_m2_terrain BETWEEN 30 AND 10000
  AND feat_m2_constructed BETWEEN 30 AND 5000

  -- 3. ROOMS FILTER (Removing likely commercial properties like hotels)
  AND feat_bedrooms BETWEEN 1 AND 10

  -- 4. BATHROOMS FILTER
  AND feat_bathrooms BETWEEN 1 AND 15

  -- 5. COHERENCE CHECK (Optional but recommended)
  -- Constructed area shouldn't be massively larger than terrain (allowing for verticality)
  AND feat_m2_constructed <= (feat_m2_terrain * 4);