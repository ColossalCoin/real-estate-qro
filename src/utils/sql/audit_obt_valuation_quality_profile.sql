/* ==========================================================================
   HOLISTIC DATA PROFILE (DATA QUALITY REPORT)
   Objective: Scan all key numeric variables in the OBT.
   Output: Summary table with health metrics (Nulls, Zeros, Distribution).
   ========================================================================== */

WITH stats AS (
  SELECT
    -- 1. TARGET (PRICE)
    'Target: Price' as variable,
    COUNT(*) as total_records,
    COUNTIF(target_price IS NULL) as null_count,
    COUNTIF(target_price = 0) as zeros,
    ROUND(MIN(target_price), 2) as min_val,
    ROUND(AVG(target_price), 2) as average,
    ROUND(STDDEV(target_price), 2) as std_dev,
    ROUND(APPROX_QUANTILES(target_price, 100)[OFFSET(50)], 2) as median, -- P50
    ROUND(MAX(target_price), 2) as max_val
  FROM `real-estate-qro.obt_listing_features.obt_listings_valuation_features`

  UNION ALL

  SELECT
    -- 2. DIMENSION (M2 CONSTRUCTED)
    'Feature: M2 Constructed',
    COUNT(*),
    COUNTIF(feat_m2_constructed IS NULL),
    COUNTIF(feat_m2_constructed = 0),
    ROUND(MIN(feat_m2_constructed), 2),
    ROUND(AVG(feat_m2_constructed), 2),
    ROUND(STDDEV(feat_m2_constructed), 2),
    ROUND(APPROX_QUANTILES(feat_m2_constructed, 100)[OFFSET(50)], 2),
    ROUND(MAX(feat_m2_constructed), 2)
  FROM `real-estate-qro.obt_listing_features.obt_listings_valuation_features`

  UNION ALL

  SELECT
    -- 3. DIMENSION (M2 TERRAIN)
    'Feature: M2 Terrain',
    COUNT(*),
    COUNTIF(feat_m2_terrain IS NULL),
    COUNTIF(feat_m2_terrain = 0),
    ROUND(MIN(feat_m2_terrain), 2),
    ROUND(AVG(feat_m2_terrain), 2),
    ROUND(STDDEV(feat_m2_terrain), 2),
    ROUND(APPROX_QUANTILES(feat_m2_terrain, 100)[OFFSET(50)], 2),
    ROUND(MAX(feat_m2_terrain), 2)
  FROM `real-estate-qro.obt_listing_features.obt_listings_valuation_features`

  UNION ALL

  SELECT
    -- 4. BEDROOMS
    'Feature: Bedrooms',
    COUNT(*),
    COUNTIF(feat_bedrooms IS NULL),
    COUNTIF(feat_bedrooms = 0),
    MIN(feat_bedrooms),
    ROUND(AVG(feat_bedrooms), 1),
    ROUND(STDDEV(feat_bedrooms), 1),
    APPROX_QUANTILES(feat_bedrooms, 100)[OFFSET(50)],
    MAX(feat_bedrooms)
  FROM `real-estate-qro.obt_listing_features.obt_listings_valuation_features`

  UNION ALL

  SELECT
    -- 5. BATHROOMS
    'Feature: Bathrooms',
    COUNT(*),
    COUNTIF(feat_bathrooms IS NULL),
    COUNTIF(feat_bathrooms = 0),
    MIN(feat_bathrooms),
    ROUND(AVG(feat_bathrooms), 1),
    ROUND(STDDEV(feat_bathrooms), 1),
    APPROX_QUANTILES(feat_bathrooms, 100)[OFFSET(50)],
    MAX(feat_bathrooms)
  FROM `real-estate-qro.obt_listing_features.obt_listings_valuation_features`
)

SELECT
  variable,
  total_records,
  null_count,
  -- Calculate health percentage
  ROUND((null_count / total_records) * 100, 1) as pct_nulls,
  zeros,
  min_val,
  median,
  average,
  max_val,
  std_dev
FROM stats
ORDER BY variable;