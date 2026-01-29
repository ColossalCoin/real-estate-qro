/* ==========================================================================
   SILVER LAYER: GEOGRAPHIC DIMENSIONS
   --------------------------------------------------------------------------
   File: 01_geo_dimensions.sql
   Objective: 
     Transform raw geographic data into native BigQuery GEOGRAPHY types.
     This establishes the spatial framework for spatial joins.
   
   Sources:
     - queretaro_staging.geo_grid_mosaics (JSONL in a raw string column)
     - queretaro_staging.geo_neighborhoods (Lat/Lon columns)
   ========================================================================== */

-- --------------------------------------------------------------------------
-- 1. PROCESS GRID / MOSAICS (The Polygons)
-- --------------------------------------------------------------------------
-- We parse the JSON string to extract the polygon geometry.
-- 'make_valid' fixes minor topological errors automatically.

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.dim_geo_grid_polygons` AS
SELECT
  -- ID Extraction: Adjust '$.properties.id' to match your specific JSON key
  -- using COALESCE to fallback to a UUID if ID is missing.
  COALESCE(
    JSON_VALUE(geo_raw, '$.properties.id'), 
    GENERATE_UUID()
  ) AS grid_id,

  -- Geometry Parsing
  ST_GEOGFROMGEOJSON(JSON_QUERY(geo_raw, '$.geometry'), make_valid => true) AS polygon_geom,

  -- Municipality Name Extraction
  UPPER(TRIM(JSON_VALUE(geo_raw, '$.properties.NAME_2'))) AS municipality_name

FROM `real-estate-qro.queretaro_staging.geo_grid_mosaics`
WHERE geo_raw IS NOT NULL;


-- --------------------------------------------------------------------------
-- 2. PROCESS NEIGHBORHOODS (The Centroids)
-- --------------------------------------------------------------------------
-- We convert numeric Latitude/Longitude columns into a GEOGRAPHY Point.

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.dim_geo_neighborhood_centroids` AS
SELECT
  -- External key
  clean_address,

  -- Create the Point: ST_GEOGPOINT(Longitude, Latitude)
  -- We employ SAFE_CAST to prevent the script from failing on non-numeric bad data
  ST_GEOGPOINT(
    SAFE_CAST(longitude AS FLOAT64),
    SAFE_CAST(latitude AS FLOAT64)
  ) AS neighborhood_point

FROM `real-estate-qro.queretaro_staging.geo_neighborhoods`
-- Filter out rows where coordinates are missing
WHERE longitude IS NOT NULL AND latitude IS NOT NULL;