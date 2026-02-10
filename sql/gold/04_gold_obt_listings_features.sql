/* ==========================================================================
   LAYER: GOLD (Data Mart)
   TABLE: obt_listings_valuation_features
   DESCRIPTION:
     This is the One Big Table (OBT) used for the Valuation Machine Learning Model.

   DATA FLOW STRATEGY:
     1. CRIME: Aggregate raw crime logs into statistics per municipality.
     2. GEO-LOCATION: Spatially map each listing (Point) to an Official Municipality (Polygon).
     3. MERGE: Combine physical house features with the calculated environmental context.
========================================================================== */

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_marts.obt_listings_valuation_features` AS

WITH

/* --------------------------------------------------------------------------
   CTE 1: CRIME CONTEXT AGGREGATION
   Goal: Convert raw crime logs into features (columns) per municipality.
   Logic: We use STRICT EQUALITY based on the official crime catalog.
-------------------------------------------------------------------------- */
crime_features AS (
  SELECT
    municipality_name,

    -- SPECIFIC CRIME FEATURES
    SUM(CASE WHEN crime_id = 4200 THEN crime_rate ELSE 0 END) AS feat_crime_residential,
    SUM(CASE WHEN crime_id = 4100 THEN crime_rate ELSE 0 END) AS feat_crime_vehicle,
    SUM(CASE WHEN crime_id = 4400 THEN crime_rate ELSE 0 END) AS feat_crime_street,
    SUM(CASE WHEN crime_id = 1100 THEN crime_rate ELSE 0 END) AS feat_crime_homicide,
    SUM(CASE WHEN crime_id = 9000 THEN crime_rate ELSE 0 END) AS feat_crime_injuries,
    SUM(CASE WHEN crime_id = 8000 THEN crime_rate ELSE 0 END) AS feat_crime_drug_dealing

  FROM `real-estate-qro.queretaro_data_warehouse.fact_context_crime`
  GROUP BY 1
),

/* --------------------------------------------------------------------------
   CTE 2: GEO-AMENITIES (DISTANCES)  <--- ¡NUEVO BLOQUE! 🌍
   Goal: Retrieve pre-calculated distances to schools, parks, etc.
   Source: The Intermediate table we created in the previous step.
-------------------------------------------------------------------------- */
geo_amenities AS (
  SELECT
    listing_id,
    dist_school,
    dist_university,
    dist_hospital,
    dist_mall,
    dist_park,
    dist_industrial
  FROM `real-estate-qro.queretaro_data_warehouse.int_geo_features_distances`
),

/* --------------------------------------------------------------------------
   CTE 3: SPATIAL MAPPING (The "Point-in-Polygon" Step)
   Goal: Identify the official municipality for every listing.
   Logic: Use BigQuery GIS functions to check which polygon contains the house.
-------------------------------------------------------------------------- */
listings_spatially_mapped AS (
  SELECT
    l.*,

    -- Extract the official Municipality Name from the Map Layer (dim_geo_grid_polygons)
    poly.municipality_name AS official_municipality_name,

  FROM `real-estate-qro.queretaro_data_warehouse.fact_listings_cleaned` l

  -- SPATIAL JOIN:
  -- Join the Listing (l) with the Map Polygon (poly)
  -- ONLY IF the Polygon geometry contains the Listing point.
  INNER JOIN `real-estate-qro.queretaro_data_warehouse.dim_geo_grid_polygons` poly
    ON ST_CONTAINS(poly.polygon_geom, l.listing_geom)
)

/* --------------------------------------------------------------------------
   FINAL SELECT: ASSEMBLY
   Goal: Bring everything together into the final OBT.
-------------------------------------------------------------------------- */
SELECT
  -- 1. IDENTIFIERS & TARGET
  l.listing_id,
  l.price_mxn AS target_price,

  -- 2. PHYSICAL PROPERTY FEATURES
  l.m2_constructed AS feat_m2_constructed,
  l.m2_terrain AS feat_m2_terrain,
  l.bedrooms AS feat_bedrooms,
  l.bathrooms AS feat_bathrooms,
  l.parking_spots AS feat_parking_spots,
  l.is_new_property AS feat_is_new,

  -- 3. AMENITIES (Boolean Flags)
  l.has_security AS feat_has_security,
  l.has_garden AS feat_has_garden,
  l.has_pool AS feat_has_pool,
  l.has_gym AS feat_has_gym,
  l.has_kitchen AS feat_has_kitchen,
  l.has_terrace AS feat_has_terrace,

  -- 4. GEOGRAPHIC CONTEXT (Derived from Spatial Join)
  l.municipality_join_key AS feat_municipality,
  l.official_neighborhood_name AS feat_neighborhood,

  -- 5. SPATIAL AMENITIES (Distances) <--- ¡NUEVAS COLUMNAS! 📏
  -- Usamos COALESCE por seguridad, aunque la tabla int_ ya debería venir limpia.
  COALESCE(g.dist_school, 5000) AS feat_dist_school,
  COALESCE(g.dist_university, 5000) AS feat_dist_university,
  COALESCE(g.dist_hospital, 5000) AS feat_dist_hospital,
  COALESCE(g.dist_mall, 5000) AS feat_dist_mall,
  COALESCE(g.dist_park, 5000) AS feat_dist_park,
  COALESCE(g.dist_industrial, 5000) AS feat_dist_industrial,

  -- 6. SAFETY CONTEXT (Derived from Crime Join)
  -- We use COALESCE(x, 0) because if a municipality has NO crime records,
  -- the Join returns NULL. We treat this as 0 crimes (safest).
  COALESCE(c.feat_crime_residential, 0) AS feat_crime_residential,
  COALESCE(c.feat_crime_vehicle, 0) AS feat_crime_vehicle,
  COALESCE(c.feat_crime_street, 0) AS feat_crime_street,
  COALESCE(c.feat_crime_homicide, 0) AS feat_crime_homicide,
  COALESCE(c.feat_crime_injuries, 0) AS feat_crime_injueries,
  COALESCE(c.feat_crime_drug_dealing, 0) AS feat_crime_drug_dealing,

  -- 7. Metadata
  CURRENT_TIMESTAMP() AS obt_created_at

FROM listings_spatially_mapped l

-- JOIN 1: CRIME (Left Join en caso de que no haya crimen registrado)
LEFT JOIN crime_features c
  ON l.official_municipality_name = c.municipality_name

-- JOIN 2: GEO AMENITIES (Left Join usando ID de listing)
LEFT JOIN geo_amenities g
  ON l.listing_id = g.listing_id;