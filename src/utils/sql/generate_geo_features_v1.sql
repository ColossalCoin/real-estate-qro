/* ==================================================================================
   FEATURE ENGINEERING: SPATIAL DISTANCES TO AMENITIES

   Description:
     Calculates the geodesic distance (in meters) from each listing to the nearest
     point of interest (POI) for various categories (schools, parks, hospitals, etc.).

   Logic & Assumptions:
     1. Search Radius: We restrict the search to a 20km (20000m) radius using ST_DWITHIN
        to optimize BigQuery geospatial computation costs.
     2. Nearest Neighbor: For each listing, we only keep the SINGLE nearest amenity
        per category.
     3. Imputation (The "5km Rule"): If no amenity of a specific type is found
        within the 20km radius, the value is imputed as 20000. This acts as a
        "distance penalty" for the model, treating missing data as "far away"
        rather than NULL.

   Input Tables:
     - dim_listings (Silver Layer)
     - dim_context_amenities (Silver Layer)

   Output Table:
     - int_geo_features_distances (Intermediate/Feature Layer)
   ================================================================================== */

CREATE OR REPLACE TABLE `real-estate-qro.queretaro_data_warehouse.int_geo_features_distances` AS

WITH
-- 1. Get Target Listings (Base Population)
-- We filter out listings without valid coordinates to avoid calculation errors.
target_listings AS (
    SELECT
        listing_id,
        listing_geom
    FROM `real-estate-qro.queretaro_data_warehouse.fact_listings_cleaned`
),

-- 2. Get Context Amenities (Points of Interest)
context_amenities AS (
    SELECT
        type,
        amenity_geom
    FROM `real-estate-qro.queretaro_data_warehouse.dim_context_amenities`
),

-- 3. Spatial Join (Optimized with ST_DWITHIN)
-- We calculate the exact distance only for amenities within the 5km buffer.
nearby_features AS (
    SELECT
        L.listing_id,
        A.type,
        ST_DISTANCE(L.listing_geom, A.amenity_geom) as distance_meters
    FROM target_listings L
    JOIN context_amenities A
    -- ST_DWITHIN leverages BigQuery's S2 geometry indexing for speed
    ON ST_DWITHIN(L.listing_geom, A.amenity_geom, 20000) -- 20km Max Radius
),

-- 4. Find Minimum Distance per Type
-- If there are 3 schools nearby, we only care about the closest one.
closest_features AS (
    SELECT
        listing_id,
        type,
        MIN(distance_meters) as min_distance
    FROM nearby_features
    GROUP BY listing_id, type
)

-- 5. Pivot and Impute (Final Assembly)
-- Transform rows (types) into columns (features) and apply the 20000m cap for nulls.
SELECT
    L.listing_id,

    -- Education
    COALESCE(MAX(CASE WHEN type = 'education_school' THEN min_distance END), 20000) AS dist_school,
    COALESCE(MAX(CASE WHEN type = 'education_university' THEN min_distance END), 20000) AS dist_university,

    -- Health
    COALESCE(MAX(CASE WHEN type = 'health_hospital' THEN min_distance END), 20000) AS dist_hospital,
    COALESCE(MAX(CASE WHEN type = 'health_local' THEN min_distance END), 20000) AS dist_health_clinic,

    -- Commercial & Hubs
    COALESCE(MAX(CASE WHEN type = 'hub_commercial' THEN min_distance END), 20000) AS dist_mall,
    COALESCE(MAX(CASE WHEN type = 'hub_industrial' THEN min_distance END), 20000) AS dist_industrial,

    -- Nature & Leisure
    COALESCE(MAX(CASE WHEN type = 'nature_park' THEN min_distance END), 20000) AS dist_park,
    COALESCE(MAX(CASE WHEN type = 'nature_green_area' THEN min_distance END), 20000) AS dist_green_area,
    COALESCE(MAX(CASE WHEN type = 'nature_playground' THEN min_distance END), 20000) AS dist_playground,

    -- Services
    COALESCE(MAX(CASE WHEN type = 'other_service' THEN min_distance END), 20000) AS dist_service,
    COALESCE(MAX(CASE WHEN type = 'shop_convenience' THEN min_distance END), 20000) AS dist_convenience,
    COALESCE(MAX(CASE WHEN type = 'shop_market' THEN min_distance END), 20000) AS dist_market,
    COALESCE(MAX(CASE WHEN type = 'shop_supermarket' THEN min_distance END), 20000) AS dist_supermarket,

    -- Tourism
    COALESCE(MAX(CASE WHEN type = 'municipal_center' THEN min_distance END), 20000) AS dist_center,
    COALESCE(MAX(CASE WHEN type = 'hub_tourism' THEN min_distance END), 20000) AS dist_tourism,

    -- Metadata
    CURRENT_TIMESTAMP() as created_at

FROM target_listings L
LEFT JOIN closest_features C ON L.listing_id = C.listing_id
GROUP BY L.listing_id;
