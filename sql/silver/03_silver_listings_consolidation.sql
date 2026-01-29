/* ==========================================================================
   SILVER LAYER: LISTINGS ENRICHED (WHITESPACE CLEANING & DEDUPLICATION)
   --------------------------------------------------------------------------
   File: 03_silver_listings_consolidation.sql
   Objective: 
     1. Clean Listings & Metrics.
     2. DEDUPLICATE NEIGHBORHOODS: Collapse " Jurica" and "Jurica" into one entry.
     3. TEXT JOIN: Link Listing to the unique Neighborhood key.
     4. SPATIAL JOIN: Link inherited geometry to Grid.
   ========================================================================== */

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.fact_listings_cleaned` AS

WITH raw_listings AS (
  SELECT
    TO_HEX(MD5(url)) AS listing_id,
    url AS original_url,
    
    -- Metrics
    price_numeric AS price_mxn,
    m2_constructed,
    m2_terrain,
    
    -- Features
    bedrooms,
    bathrooms,
    parking_spots,
    has_security,
    has_garden,
    has_pool,
    has_terrace,
    has_gym,
    is_new_property,
    has_kitchen,
    description,
    
    -- JOIN KEY PREPARATION
    UPPER(TRIM(clean_address)) AS join_address_key,
    
    COALESCE(extraction_date, CURRENT_TIMESTAMP()) as processed_date
  FROM `real-estate-qro.queretaro_staging.listings_enriched`
),

-- We select distinct neighborhoods by their TRIMMED name.
deduplicated_neighborhoods AS (
  SELECT 
    -- This becomes our unique joining key (e.g., "JURICA")
    UPPER(TRIM(clean_address)) AS unique_key,
    
    -- We take the cleanest display version (removing surrounding spaces)
    ANY_VALUE(TRIM(clean_address)) AS official_neighborhood_name,
    
    -- We assume the centroid is practically the same for both versions
    ANY_VALUE(neighborhood_point) AS neighborhood_point
    
  FROM `real-estate-qro.queretaro_data_warehouse.dim_geo_neighborhood_centroids`
  WHERE clean_address IS NOT NULL
  -- This GROUP BY is what eliminates the duplicates shown in your screenshot
  GROUP BY 1
)

SELECT
  l.listing_id,
  l.original_url,
  l.price_mxn,
  l.m2_constructed,
  l.m2_terrain,
  l.bedrooms,
  l.bathrooms,
  l.parking_spots,
  l.has_security,
  l.has_garden,
  l.has_pool,
  l.has_terrace,
  l.has_gym,
  l.is_new_property,
  l.has_kitchen,
  l.description,
  
  -- Now we get the clean, trimmed name (e.g., "Jurica" without spaces)
  n.official_neighborhood_name,
  n.neighborhood_point AS listing_geom,
  
  g.grid_id,
  
  l.processed_date

FROM raw_listings l

-- JOIN 1: Connect to the deduplicated dictionary of neighborhoods
INNER JOIN deduplicated_neighborhoods n
  ON l.join_address_key = n.unique_key

-- JOIN 2: Spatial Grid
INNER JOIN `real-estate-qro.queretaro_data_warehouse.dim_geo_grid_polygons` g
  ON ST_CONTAINS(g.grid_geom, n.neighborhood_point)

-- FINAL SAFETY NET (Optional but recommended)
-- Just in case a point falls on a Grid border line
QUALIFY ROW_NUMBER() OVER(PARTITION BY l.listing_id ORDER BY g.grid_id) = 1;