# Multi-Sensor Fusion of Landsat and Sentinel-1 SAR for 
# LULC Change Mapping and LST Projection — Dhaka, Bangladesh

## Description
Google Earth Engine script for spatiotemporal LULC change 
mapping and Land Surface Temperature projection in Dhaka, 
Bangladesh (2015–2025–2035) using Landsat 8/9 optical imagery 
and Sentinel-1 C-band SAR fusion.

## Study Area
Dhaka District, Bangladesh (328.2 km²)
Centroid: 23.76°N, 90.40°E

## Data Sources
- Landsat 8/9 Collection 2 Level-2 (USGS)
- Sentinel-1 GRD IW C-band (ESA/Copernicus)
- GHSL P2023A GHS_BUILT_S (JRC)
- All accessed via Google Earth Engine

## Methods
- Random Forest LULC classification (optical-only vs SAR-fused)
- Gradient Boosted Trees LST regression
- Transition RF model for 2035 LULC projection
- Spatial holdout validation (300m grid)

## Requirements
- Google Earth Engine account
- BGD_adm3 boundary asset uploaded to your GEE assets

## Author
Newaz Ibrahim Khan  
BSc Computer Science and Engineering  
World University of Bangladesh

## Citation
Khan, N.I. (2026). Multi-Sensor Fusion of Landsat and 
Sentinel-1 SAR for Spatiotemporal LULC Change Mapping and 
LST Projection in Dhaka. Geocarto International.
