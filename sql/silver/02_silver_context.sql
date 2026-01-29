/* ==========================================================================
   SILVER LAYER: CONTEXT DATA (CRIME & AMENITIES)
   --------------------------------------------------------------------------
   File: 02_silver_context.sql
   Objective: 
     Clean, standardize, and geolocate context datasets.
     These tables will be used to calculate "distance to X" and "risk in area".
   
   Sources:
     - queretaro_staging.context_amenities (OSM Data)
     - queretaro_staging.context_crime (ONC/Security Data)
   ========================================================================== */

-- --------------------------------------------------------------------------
-- 1. PROCESS AMENITIES
-- --------------------------------------------------------------------------
-- Filter only useful amenities and create GEOGRAPHY Points.

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.dim_context_amenities` AS
SELECT
  GENERATE_UUID() AS amenity_id,
  
  LOWER(TRIM(category)) AS type,       -- e.g. school, hospital, park
  INITCAP(TRIM(name)) AS place_name,
  ST_GEOGPOINT(
    SAFE_CAST(longitude AS FLOAT64), 
    SAFE_CAST(latitude AS FLOAT64)
  ) AS amenity_geom

FROM `real-estate-qro.queretaro_staging.context_amenities`
WHERE longitude IS NOT NULL AND latitude IS NOT NULL AND category IS NOT NULL;


-- --------------------------------------------------------------------------
-- 2. PROCESS CRIME
-- --------------------------------------------------------------------------
-- Create GEOGRAPHY Points for each crime event to allow spatial aggregation.

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.fact_context_crime` AS
SELECT
  GENERATE_UUID() AS crime_id,
  
  UPPER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                 CASE WHEN UPPER(municipio) LIKE '%BONFIL%' THEN 'AMEALCO DE BONFIN'
                      ELSE TRIM(municipio)
                 END,
    'Á', 'A'), 'É', 'E'), 'Í', 'I'), 'Ó', 'O'), 'Ú', 'U'), 'Ñ', 'N')
  ) AS municipality_join_key,
  delito AS crime_type,
  SAFE_CAST(tasa AS FLOAT64) AS crime_rate,
  PARSE_DATE('%Y-%m-%d', CONCAT(fecha, '-01')) AS reference_date

FROM `real-estate-qro.queretaro_staging.context_crime`
WHERE municipio IS NOT NULL;