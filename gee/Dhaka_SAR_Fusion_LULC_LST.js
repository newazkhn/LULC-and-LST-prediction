// ============================================================
//  Multi-Sensor Fusion of Landsat and Sentinel-1 SAR for
//  LULC Change Mapping and LST Projection — Dhaka, Bangladesh
//  Study period : 2015 (observed) | 2025 (observed) | 2035 (projected)
//
//  Author  : Newaz Ibrahim Khan
//  Journal : Geocarto International (submitted)
//
//  What this script does:
//    1. Loads and pre-processes Landsat 8/9 optical imagery
//    2. Loads and pre-processes Sentinel-1 C-band SAR imagery
//    3. Derives emissivity predictors (FVC, LSE) and loads GHSL density
//    4. Trains a Random Forest LULC classifier (optical-only vs SAR-fused)
//    5. Projects LULC to 2035 using a transition Random Forest model
//    6. Trains a Gradient Boosted Trees LST regressor
//    7. Projects LST to 2035 using class-mean predictor substitution
//    8. Validates both models using a spatial holdout design
//    9. Exports all maps, CSVs, and feature importance data
//
//  Required imports (set in GEE script editor):
//    - table       : BGD administrative boundaries (FeatureCollection, NAME_3)
//    - water       : Water training polygons (property 'Class' = 0)
//    - built_up    : Built-Up training polygons (property 'Class' = 1)
//    - green_area  : Green Area training polygons (property 'Class' = 2)
// ============================================================


// ============================================================
//  SECTION 0 — GLOBAL SETTINGS
// ============================================================

var ROI_NAME      = 'Tejgaon';        // NAME_3 value in the admin boundary table
var PIXEL_SIZE    = 30;               // Analysis resolution (metres) — matches Landsat
var RANDOM_SEED   = 7;                // Fixed seed for reproducible train/test splits
var EXPORT_CRS    = 'EPSG:32646';     // UTM Zone 46N — appropriate for Dhaka
var EXPORT_FOLDER = 'Tejgaon_Maps';   // Google Drive folder for all outputs


// ============================================================
//  SECTION 1 — STUDY AREA
// ============================================================

// Filter the admin boundary table to extract the Tejgaon polygon
var roiFeature = table.filter(ee.Filter.eq('NAME_3', ROI_NAME));
var roi        = roiFeature.geometry();

print('Admin boundary check:', table.limit(5));
Map.addLayer(roiFeature, {}, 'Study Area Boundary');
Map.centerObject(roiFeature, 12);


// ============================================================
//  SECTION 2 — CLASS DEFINITIONS
// ============================================================

// Three land cover classes — consistent throughout the script
// 0 = Water | 1 = Built-Up | 2 = Green Area
var CLASS_NAMES   = ee.Dictionary({ 0: 'Water', 1: 'Built_Up', 2: 'Green_area' });
var CLASS_PALETTE = ['#0c2c84', '#dfff0b', '#008000'];  // dark blue, yellow, green


// ============================================================
//  SECTION 3 — LANDSAT PRE-PROCESSING
// ============================================================

// Cloud and shadow masking using the QA_PIXEL bitmask
// Bits tested: 1=dilated cloud, 2=cirrus, 3=cloud, 4=cloud shadow
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 3).eq(0)   // no cloud
               .and(qa.bitwiseAnd(1 << 4).eq(0))   // no cloud shadow
               .and(qa.bitwiseAnd(1 << 2).eq(0))   // no cirrus
               .and(qa.bitwiseAnd(1 << 1).eq(0));  // no dilated cloud
  return image.updateMask(clear);
}

// Build a cloud-free composite for a given date window and sensor
// SR bands → median composite (robust against residual cloud edges)
// LST band → mean composite (preserves real thermal extremes)
//
// Inter-sensor note: Landsat 8 (LC08) and Landsat 9 (LC09) are
// cross-calibrated by USGS to within <0.12 K (Roy et al. 2021),
// so they can be treated as equivalent for this analysis.
function buildLandsatComposite(startDate, endDate, sensorCollection) {
  var collection = ee.ImageCollection(sensorCollection)
    .filterDate(startDate, endDate)
    .filterBounds(roi)
    .map(maskClouds)
    .map(function(image) {

      // Convert DN to surface reflectance: DN × 0.0000275 − 0.2
      var sr = image.select('SR_B.*').multiply(0.0000275).add(-0.2);

      // Spectral indices
      var ndvi  = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
      var ndbi  = sr.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
      var mndwi = sr.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');

      // Convert thermal band to Celsius: DN × 0.00341802 + 149.0 − 273.15
      var lst = image.select('ST_B10')
        .multiply(0.00341802).add(149.0).subtract(273.15)
        .rename('LST');

      return sr.addBands([ndvi, ndbi, mndwi, lst]);
    });

  return collection.select(['SR_B.*', 'NDVI', 'NDBI', 'MNDWI']).median()
    .addBands(collection.select('LST').mean())
    .clip(roi);
}


// ============================================================
//  SECTION 4 — SENTINEL-1 SAR FEATURE ENGINEERING
// ============================================================

// Builds a 6-band SAR feature stack for a given date window:
//   VV_dB, VH_dB, VV−VH ratio, VV entropy, VV contrast, VV variance
//
// Design choices:
//   - Descending pass only → consistent radar look geometry over Dhaka
//   - dB conversion → normalises the right-skewed backscatter distribution
//   - Median composite → reduces single-scene speckle noise
//   - GLCM textures on dB values → dB quantises more uniformly than linear
//   - Bilinear resampling to 30 m → spatial consistency with Landsat
//
// Radar shadow note:
//   Tall buildings create low-backscatter shadow zones that resemble water.
//   Mitigation: (a) multi-scene median attenuates geometry-dependent shadows,
//   (b) optical bands dominate the RF classifier where SAR is ambiguous,
//   (c) VV entropy distinguishes shadow (uniform) from water (MNDWI > 0).
//   No DEM-derived shadow mask was applied; shadow extent at 30 m after
//   resampling is expected to be sub-pixel for typical Dhaka building heights.

function buildSARFeatures(startDate, endDate, targetCRS) {

  // Load Sentinel-1 IW GRD, descending pass, dual-polarisation
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .select(['VV', 'VH']);

  print('Sentinel-1 scenes loaded (' + startDate + '):', s1.size());

  // Convert linear backscatter to dB: σ°dB = 10 × log10(σ°linear)
  var s1dB = s1.map(function(img) {
    return ee.Image(10).multiply(img.log10())
      .rename(['VV_dB', 'VH_dB'])
      .copyProperties(img, img.propertyNames());
  }).median().clip(roi);

  // Co-polarisation ratio band: high values → urban double-bounce
  var vvMinusVh = s1dB.select('VV_dB')
    .subtract(s1dB.select('VH_dB'))
    .rename('VV_minus_VH');

  // GLCM texture features from VV channel
  // Quantise to integer first (×10 preserves 0.1 dB precision)
  var vvInt = s1dB.select('VV_dB').multiply(10).int32();
  var glcm  = vvInt.glcmTexture({ size: 3 });

  var vvEntropy  = glcm.select('VV_dB_sent').rename('VV_entropy');
  var vvContrast = glcm.select('VV_dB_contrast').rename('VV_contrast');
  var vvVariance = glcm.select('VV_dB_var').rename('VV_variance');

  // Assemble stack and resample to Landsat 30 m grid
  return s1dB
    .addBands([vvMinusVh, vvEntropy, vvContrast, vvVariance])
    .float()
    .resample('bilinear')
    .reproject({ crs: targetCRS, scale: PIXEL_SIZE })
    .unmask(0);
}


// ============================================================
//  SECTION 5 — EMISSIVITY AND GHSL PREDICTORS
// ============================================================

// Fractional Vegetation Cover (FVC) from NDVI
// Method: Carlson & Ripley (1997)
// FVC = ((NDVI − NDVI_soil) / (NDVI_veg − NDVI_soil))²
// NDVI_soil = 0.05 (bare urban hardscape)
// NDVI_veg  = 0.86 (dense tropical canopy)
//
// Land Surface Emissivity (LSE) from FVC
// Method: Sobrino et al. (2004), calibrated for Landsat TIR Band 10
// LSE = 0.004 × FVC + 0.986
//
// These bands give the GBT regressor explicit emissivity information,
// reducing underprediction errors at thermally extreme pixels (metal
// rooftops, shaded water bodies) that optical reflectance cannot resolve.

function addEmissivityBands(composite) {
  var ndvi = composite.select('NDVI');
  var fvc  = ndvi.subtract(0.05).divide(0.86 - 0.05).pow(2)
               .clamp(0, 1).rename('FVC');
  var lse  = fvc.multiply(0.004).add(0.986).rename('LSE');
  return composite.addBands([fvc, lse]);
}

// GHSL built-up surface fraction (JRC P2023A, 2020 epoch)
// Raw values are built-up area (m²) per 100 m cell → divide by 10,000 → fraction 0–1
// Resampled to 30 m Landsat grid via bilinear interpolation
// Captures urban morphological density not encoded by spectral or SAR bands
var ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S')
  .filterDate('2020-01-01', '2021-01-01')
  .first()
  .select('built_surface')
  .divide(10000)
  .clamp(0, 1)
  .rename('GHSL_built')
  .resample('bilinear')
  .reproject({ crs: 'EPSG:32646', scale: 30 })
  .clip(roi);

print('GHSL band check:', ghsl.bandNames());


// ============================================================
//  SECTION 6 — PREDICTOR BAND LISTS
// ============================================================

// Optical-only LULC predictor set (9 bands)
var OPTICAL_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                     'NDVI', 'NDBI', 'MNDWI'];

// SAR bands (6 bands)
var SAR_BANDS = ['VV_dB', 'VH_dB', 'VV_minus_VH',
                 'VV_entropy', 'VV_contrast', 'VV_variance'];

// Fused LULC predictor set (15 bands)
var FUSED_LULC_BANDS = OPTICAL_BANDS.concat(SAR_BANDS);

// Auxiliary LST predictor bands
var AUX_BANDS = ['FVC', 'LSE', 'GHSL_built'];

// LST predictor sets (optical-only and fused — both include emissivity + GHSL)
var OPT_LST_BANDS    = OPTICAL_BANDS.concat(AUX_BANDS).concat(['lc']);
var FUSED_LST_BANDS  = OPTICAL_BANDS.concat(SAR_BANDS).concat(AUX_BANDS).concat(['lc']);


// ============================================================
//  SECTION 7 — BUILD COMPOSITES AND DIAGNOSTIC CHECKS
// ============================================================

// April acquisition windows — pre-monsoon LST peak, minimum cloud cover
// Landsat 8 used for 2015, Landsat 9 for 2025 (both Collection 2 Level-2)
var WINDOW_2015 = { start: '2015-04-01', end: '2015-04-30',
                    sensor: 'LANDSAT/LC08/C02/T1_L2' };
var WINDOW_2025 = { start: '2025-04-01', end: '2025-04-30',
                    sensor: 'LANDSAT/LC09/C02/T1_L2' };

// Diagnostic: print acquisition counts and dates
// If either year returns 0, widen the window to April–May
var diag2015 = ee.ImageCollection(WINDOW_2015.sensor)
  .filterDate(WINDOW_2015.start, WINDOW_2015.end)
  .filterBounds(roi).map(maskClouds);
var diag2025 = ee.ImageCollection(WINDOW_2025.sensor)
  .filterDate(WINDOW_2025.start, WINDOW_2025.end)
  .filterBounds(roi).map(maskClouds);

print('Clear acquisitions — 2015:', diag2015.size());
print('Clear acquisitions — 2025:', diag2025.size());
print('Acquisition dates — 2015:',
  diag2015.aggregate_array('system:time_start')
    .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); }));
print('Acquisition dates — 2025:',
  diag2025.aggregate_array('system:time_start')
    .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); }));

// Build Landsat composites and add emissivity + GHSL bands
var composite2015 = addEmissivityBands(
  buildLandsatComposite(WINDOW_2015.start, WINDOW_2015.end, WINDOW_2015.sensor)
).addBands(ghsl);

var composite2025 = addEmissivityBands(
  buildLandsatComposite(WINDOW_2025.start, WINDOW_2025.end, WINDOW_2025.sensor)
).addBands(ghsl);

print('2025 composite bands:', composite2025.bandNames());

// Build SAR feature stacks — use explicit CRS string (not lazy projection)
var CRS = 'EPSG:32646';
var sar2015 = buildSARFeatures(WINDOW_2015.start, WINDOW_2015.end, CRS);
var sar2025 = buildSARFeatures(WINDOW_2025.start, WINDOW_2025.end, CRS);

print('SAR band names:', sar2025.bandNames());

// Fused images (Landsat + SAR)
var fused2015 = composite2015.addBands(sar2015);
var fused2025 = composite2025.addBands(sar2025);

print('Fused 2025 bands (should be 20+):', fused2025.bandNames());

// Display composites for visual inspection
Map.addLayer(composite2015, { bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3 },
             'True Colour 2015');
Map.addLayer(composite2025, { bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3 },
             'True Colour 2025');
Map.addLayer(sar2025, { bands: ['VV_dB', 'VH_dB', 'VV_minus_VH'], min: -20, max: 0 },
             'SAR False Colour 2025 (VV=R, VH=G, VV-VH=B)');


// ============================================================
//  SECTION 8 — LULC CLASSIFICATION (ABLATION STUDY)
// ============================================================

// Merge all training polygons into one FeatureCollection
var trainingPolygons = water.merge(built_up).merge(green_area);

// --- Optical-only classifier ---
var samplesOpt = composite2025.select(OPTICAL_BANDS).sampleRegions({
  collection: trainingPolygons, properties: ['Class'], scale: PIXEL_SIZE
});
var splitOpt   = samplesOpt.randomColumn('random', RANDOM_SEED);
var trainOpt   = splitOpt.filter(ee.Filter.lt('random', 0.7));
var testOpt    = splitOpt.filter(ee.Filter.gte('random', 0.7));

var classifierOpt = ee.Classifier.smileRandomForest({
  numberOfTrees: 300, minLeafPopulation: 2, bagFraction: 0.7, seed: RANDOM_SEED
}).train({ features: trainOpt, classProperty: 'Class', inputProperties: OPTICAL_BANDS });

var cmOpt = testOpt.classify(classifierOpt).errorMatrix('Class', 'classification');
print('--- OPTICAL-ONLY LULC ACCURACY ---');
print('Confusion matrix:', cmOpt);
print('Overall Accuracy:', cmOpt.accuracy());
print('Kappa coefficient:', cmOpt.kappa());

// --- SAR-fused classifier ---
var samplesFused = fused2025.select(FUSED_LULC_BANDS).sampleRegions({
  collection: trainingPolygons, properties: ['Class'], scale: PIXEL_SIZE
});
var splitFused  = samplesFused.randomColumn('random', RANDOM_SEED);
var trainFused  = splitFused.filter(ee.Filter.lt('random', 0.7));
var testFused   = splitFused.filter(ee.Filter.gte('random', 0.7));

var classifierFused = ee.Classifier.smileRandomForest({
  numberOfTrees: 300, minLeafPopulation: 2, bagFraction: 0.7, seed: RANDOM_SEED
}).train({ features: trainFused, classProperty: 'Class', inputProperties: FUSED_LULC_BANDS });

var cmFused = testFused.classify(classifierFused).errorMatrix('Class', 'classification');
print('--- SAR-FUSED LULC ACCURACY ---');
print('Confusion matrix:', cmFused);
print('Overall Accuracy:', cmFused.accuracy());
print('Kappa coefficient:', cmFused.kappa());
print('Feature importances (fused RF):', classifierFused.explain());

// Use the fused classifier for all subsequent mapping
var activeClassifier = classifierFused;


// ============================================================
//  SECTION 9 — APPLY CLASSIFIER: 2015 AND 2025 MAPS
// ============================================================

var lc2015 = fused2015.select(FUSED_LULC_BANDS).classify(activeClassifier)
               .rename('classification');
var lc2025 = fused2025.select(FUSED_LULC_BANDS).classify(activeClassifier)
               .rename('classification');

Map.addLayer(lc2015, { min: 0, max: 2, palette: CLASS_PALETTE }, 'LULC 2015');
Map.addLayer(lc2025, { min: 0, max: 2, palette: CLASS_PALETTE }, 'LULC 2025');


// ============================================================
//  SECTION 10 — TEMPORAL VALIDATION (2015 SPECTRAL STATIONARITY)
// ============================================================

// The classifier was trained on 2025 data and applied to 2015 imagery.
// This test checks whether the cross-year application is justified.
// The same training polygons are used because they represent stable surface
// types (core water bodies, dense buildings, mature canopy). Pixels that
// changed class between 2015 and 2025 will appear as confusion — making
// this a conservative test that slightly penalises 2015 accuracy.
// A drop of <5 pp OA is acceptable; a larger drop requires acknowledgement
// in the Limitations section.

var samples2015 = fused2015.select(FUSED_LULC_BANDS).sampleRegions({
  collection: trainingPolygons, properties: ['Class'], scale: PIXEL_SIZE
});
var test2015 = samples2015.randomColumn('random', RANDOM_SEED)
                 .filter(ee.Filter.gte('random', 0.7));
var cm2015   = test2015.classify(activeClassifier).errorMatrix('Class', 'classification');

print('--- TEMPORAL VALIDATION: 2025 classifier on 2015 imagery ---');
print('Confusion matrix:', cm2015);
print('Overall Accuracy (2015):', cm2015.accuracy());
print('Kappa (2015):', cm2015.kappa());
print('Compare OA to 2025 result. Drop > 5 pp requires limitations note.');


// ============================================================
//  SECTION 11 — LULC AREA CALCULATION AND CHARTS
// ============================================================

function computeLULCArea(classifiedImage) {
  var hist = classifiedImage.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: roi, scale: PIXEL_SIZE, maxPixels: 1e13
  });
  return ee.Dictionary(hist.get('classification'));
}

function printAreaChart(areaDict, label) {
  var fc = ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
    c = ee.Number(c);
    var areaKm2 = ee.Number(areaDict.get(c.format(), 0)).multiply(0.0009);
    return ee.Feature(null, { class_code: c, class: CLASS_NAMES.get(c), area_km2: areaKm2 });
  }));
  print(ui.Chart.feature.byFeature(fc, 'class', ['area_km2'])
    .setChartType('ColumnChart')
    .setOptions({
      title: 'LULC Area — ' + label,
      hAxis: { title: 'Land Cover Class' },
      vAxis: { title: 'Area (km²)' },
      legend: { position: 'none' },
      colors: CLASS_PALETTE
    }));
  return fc;
}

var area2015 = computeLULCArea(lc2015);
var area2025 = computeLULCArea(lc2025);
printAreaChart(area2015, '2015 (observed)');
printAreaChart(area2025, '2025 (observed)');


// ============================================================
//  SECTION 12 — LULC PROJECTION TO 2035
// ============================================================

// A transition Random Forest learns the spatial pattern of land cover
// change between 2015 and 2025, then extrapolates it one decade forward.
// Including 'lc_prev' (prior class) explicitly encodes land cover persistence
// and class-specific conversion probabilities.
// This is a pattern-extrapolation under business-as-usual — not a causal model.

var transitionBands = FUSED_LULC_BANDS.concat(['lc_prev']);

var transitionTrainImage = fused2015.select(FUSED_LULC_BANDS)
  .addBands(lc2015.rename('lc_prev'))
  .addBands(lc2025.rename('label'));

var transitionSamples = transitionTrainImage.sample({
  region: roi, scale: PIXEL_SIZE, numPixels: 8000, tileScale: 4, seed: 1, geometries: false
});

var transitionClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 300, bagFraction: 0.7, seed: 1
}).train({ features: transitionSamples, classProperty: 'label',
           inputProperties: transitionBands });

// Apply transition model to 2025 predictors to get 2035 projection
var lc2035 = fused2025.select(FUSED_LULC_BANDS)
  .addBands(lc2025.rename('lc_prev'))
  .classify(transitionClassifier)
  .rename('classification');

Map.addLayer(lc2035, { min: 0, max: 2, palette: CLASS_PALETTE }, 'LULC 2035 (projected)');
var area2035 = computeLULCArea(lc2035);
printAreaChart(area2035, '2035 (projected)');


// ============================================================
//  SECTION 13 — LST BASE LAYERS AND HELPER FUNCTIONS
// ============================================================

var lst2015 = composite2015.select('LST').rename('LST_2015');
var lst2025 = composite2025.select('LST').rename('LST_2025');

// Compute ROI-wide mean LST for a given image and band name
function roiMean(image, bandName) {
  return ee.Number(image.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: roi, scale: PIXEL_SIZE, maxPixels: 1e13
  }).get(bandName));
}

// Convert a FeatureCollection with class_code and mean_val to an ee.Dictionary
function fcToDict(fc) {
  var list = ee.List(fc.reduceColumns(
    ee.Reducer.toList(2), ['class_code', 'mean_val']).get('list'));
  return ee.Dictionary(list.iterate(function(pair, acc) {
    pair = ee.List(pair);
    return ee.Dictionary(acc).set(ee.Number(pair.get(0)).format(), pair.get(1));
  }, ee.Dictionary({})));
}

// Compute per-class mean of a band from the 2025 composite
function classMean(bandName) {
  return ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
    c = ee.Number(c);
    var mean = composite2025.select(bandName)
      .updateMask(lc2025.eq(c))
      .reduceRegion({ reducer: ee.Reducer.mean(), geometry: roi,
                      scale: PIXEL_SIZE, maxPixels: 1e13 })
      .get(bandName);
    return ee.Feature(null, { class_code: c, mean_val: mean });
  }));
}

// Synthesise an image where each pixel gets the class-mean value
// for a given band, based on the projected 2035 LULC map
function syntheticFromClass(lcImage, classMeanDict, outputName) {
  lcImage = lcImage.rename('lc');
  return ee.Image(0)
    .where(lcImage.eq(0), ee.Number(classMeanDict.get('0')))
    .where(lcImage.eq(1), ee.Number(classMeanDict.get('1')))
    .where(lcImage.eq(2), ee.Number(classMeanDict.get('2')))
    .rename(outputName)
    .clip(roi);
}

// Print a grouped bar chart of mean LST by class for a given epoch
function lstByClassChart(lcImage, lstImage, yearLabel) {
  var lstBand = ee.String(lstImage.bandNames().get(0));
  var fc = ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
    c = ee.Number(c);
    var mean = lstImage.updateMask(lcImage.eq(c))
      .reduceRegion({ reducer: ee.Reducer.mean(), geometry: roi,
                      scale: PIXEL_SIZE, maxPixels: 1e13 })
      .get(lstBand);
    return ee.Feature(null, { year: yearLabel, class_code: c,
                              class: CLASS_NAMES.get(c), mean_LST: mean });
  }));
  print(ui.Chart.feature.byFeature(fc, 'class', ['mean_LST'])
    .setChartType('ColumnChart')
    .setOptions({
      title: 'Mean LST by Class — ' + yearLabel,
      hAxis: { title: 'Class' }, vAxis: { title: '°C' }, legend: { position: 'none' }
    }));
  return fc;
}


// ============================================================
//  SECTION 14 — 2035 PREDICTOR SYNTHESIS
// ============================================================

// Synthesise spectral index values for 2035 from per-class means
// This ensures all predictors are consistent with the projected LULC
var ndviMeans  = fcToDict(classMean('NDVI'));
var ndbiMeans  = fcToDict(classMean('NDBI'));
var mndwiMeans = fcToDict(classMean('MNDWI'));

var ndvi2035  = syntheticFromClass(lc2035, ndviMeans,  'NDVI');
var ndbi2035  = syntheticFromClass(lc2035, ndbiMeans,  'NDBI');
var mndwi2035 = syntheticFromClass(lc2035, mndwiMeans, 'MNDWI');


// ============================================================
//  SECTION 15 — GBT LST REGRESSION MODEL
// ============================================================

// Gradient Boosted Trees (GBT) for LST regression
// Improvements over the RF baseline:
//   1. GBT algorithm — iterative residual correction reduces systematic bias
//      at thermally extreme pixels (Friedman 2001)
//   2. FVC + LSE emissivity bands — explicit emissivity information
//      (Carlson & Ripley 1997; Sobrino et al. 2004)
//   3. GHSL built-up density — urban morphology not captured by reflectance
//   4. 15,000 training samples — better coverage of thermal distribution tails
// Training target: Landsat ST_B10 LST (30 m direct thermal observation)

// Optical-only GBT (ablation baseline)
var lstTrainOpt = composite2025.select(OPTICAL_BANDS)
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .addBands(lst2025.rename('LST'));

var lstSamplesOpt = lstTrainOpt.sample({
  region: roi, scale: PIXEL_SIZE, numPixels: 15000, tileScale: 8, seed: 2, geometries: false
}).filter(ee.Filter.notNull(['LST']));

var lstRegressorOpt = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 300, shrinkage: 0.05, samplingRate: 0.7,
  maxNodes: 8, loss: 'LeastSquares', seed: 2
}).setOutputMode('REGRESSION')
  .train({ features: lstSamplesOpt, classProperty: 'LST',
           inputProperties: OPT_LST_BANDS });

// SAR-fused GBT
var lstTrainFused = fused2025.select(OPTICAL_BANDS.concat(SAR_BANDS))
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .addBands(lst2025.rename('LST'));

var lstSamplesFused = lstTrainFused.sample({
  region: roi, scale: PIXEL_SIZE, numPixels: 15000, tileScale: 8, seed: 2, geometries: false
}).filter(ee.Filter.notNull(['LST']));

var lstRegressorFused = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 300, shrinkage: 0.05, samplingRate: 0.7,
  maxNodes: 8, loss: 'LeastSquares', seed: 2
}).setOutputMode('REGRESSION')
  .train({ features: lstSamplesFused, classProperty: 'LST',
           inputProperties: FUSED_LST_BANDS });

print('Fused GBT trained. Feature importances:', lstRegressorFused.explain());
print('Optical GBT sample count:', lstSamplesOpt.size());
print('Fused GBT sample count:', lstSamplesFused.size());


// ============================================================
//  SECTION 16 — LST PROJECTION TO 2035
// ============================================================

// All predictors are synthesised from 2035 LULC class-means to ensure
// physical consistency. A pixel transitioning from Green to Built-Up
// receives Built-Up spectral and SAR signatures — not its 2025 values.
// Limitation: class-mean substitution suppresses within-class spatial
// heterogeneity. The 2035 map reflects mean class thermal behaviour,
// appropriate for trend analysis but not sub-class pattern interpretation.

// Helper to synthesise any single band from 2035 LULC class means
function syntheticBand2035(bandName, sourceImage, outputName) {
  var means = fcToDict(ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
    c = ee.Number(c);
    var mean = sourceImage.select(bandName)
      .updateMask(lc2025.eq(c))
      .reduceRegion({ reducer: ee.Reducer.mean(), geometry: roi,
                      scale: PIXEL_SIZE, maxPixels: 1e13 })
      .get(bandName);
    return ee.Feature(null, { class_code: c, mean_val: mean });
  })));
  return syntheticFromClass(lc2035, means, outputName);
}

// Synthesise all SR reflectance bands
var SR_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
var sr2035 = SR_BANDS.reduce(function(stack, b) {
  return ee.Image(stack).addBands(syntheticBand2035(b, composite2025, b));
}, ee.Image([]));

// Synthesise all SAR bands
var sarSynth2035 = SAR_BANDS.reduce(function(stack, b) {
  return ee.Image(stack).addBands(syntheticBand2035(b, sar2025, b));
}, ee.Image([]));

// Synthesise GHSL density
var ghsl2035 = syntheticBand2035('GHSL_built', composite2025, 'GHSL_built');

// Recompute emissivity from synthetic 2035 NDVI
var fvc2035 = ndvi2035.subtract(0.05).divide(0.86 - 0.05).pow(2)
                .clamp(0, 1).rename('FVC');
var lse2035 = fvc2035.multiply(0.004).add(0.986).rename('LSE');

// Assemble the fully consistent 2035 predictor stack
var predictors2035 = ee.Image(sr2035)
  .addBands([ndvi2035, ndbi2035, mndwi2035])
  .addBands(ee.Image(sarSynth2035))
  .addBands(fvc2035).addBands(lse2035).addBands(ghsl2035)
  .addBands(lc2035.rename('lc'));

var lst2035 = predictors2035.select(FUSED_LST_BANDS)
  .classify(lstRegressorFused)
  .rename('LST_2035');

// Display all three LST maps
var lstVis = { min: 28, max: 46, palette: ['blue', 'cyan', 'yellow', 'orange', 'red'] };
Map.addLayer(lst2015, lstVis, 'LST 2015 (observed)');
Map.addLayer(lst2025, lstVis, 'LST 2025 (observed)');
Map.addLayer(lst2035, lstVis, 'LST 2035 (projected)');

// Print and chart mean LST across epochs
print('Mean LST 2015 (°C):', roiMean(lst2015, 'LST_2015'));
print('Mean LST 2025 (°C):', roiMean(lst2025, 'LST_2025'));
print('Mean LST 2035 (°C):', roiMean(lst2035, 'LST_2035'));

var lstMeanFC = ee.FeatureCollection([
  ee.Feature(null, { year: '2015', mean_LST: roiMean(lst2015, 'LST_2015') }),
  ee.Feature(null, { year: '2025', mean_LST: roiMean(lst2025, 'LST_2025') }),
  ee.Feature(null, { year: '2035', mean_LST: roiMean(lst2035, 'LST_2035') })
]);
print(ui.Chart.feature.byFeature(lstMeanFC, 'year', 'mean_LST')
  .setChartType('ColumnChart')
  .setOptions({ title: 'Mean LST Over Time', hAxis: { title: 'Year' },
                vAxis: { title: '°C' }, legend: { position: 'none' } }));

var lstByClass2015 = lstByClassChart(lc2015, lst2015, '2015');
var lstByClass2025 = lstByClassChart(lc2025, lst2025, '2025');
var lstByClass2035 = lstByClassChart(lc2035, lst2035, '2035');


// ============================================================
//  SECTION 17 — LST SPATIAL HOLDOUT VALIDATION
// ============================================================

// A 300 m grid partitions the ROI into spatially independent
// training (70%) and test (30%) zones. This prevents spatial
// autocorrelation between adjacent pixels from inflating accuracy.
// Both fused and optical-only GBT models are evaluated against
// observed Landsat ST_B10 LST in the test zone.

print('--- LST SPATIAL HOLDOUT VALIDATION ---');

// Build the grid and split into train / test zones
var grid = roi.bounds()
  .coveringGrid(ee.Projection('EPSG:32646').atScale(300), 300)
  .filterBounds(roi)
  .randomColumn('rand', 42);

var trainGrid = grid.filter(ee.Filter.lt('rand', 0.7));
var testGrid  = grid.filter(ee.Filter.gte('rand', 0.7));

var trainZone = trainGrid.geometry().intersection(roi, ee.ErrorMargin(1));
var testZone  = testGrid.geometry().intersection(roi, ee.ErrorMargin(1));

print('Train zone area (m²):', trainZone.area(1));
print('Test zone area (m²):', testZone.area(1));
print('Train grid cells:', trainGrid.size());
print('Test grid cells:', testGrid.size());

Map.addLayer(ee.Image(0).paint(ee.FeatureCollection([ee.Feature(trainZone)]), 1),
             { palette: ['#2196F3'], opacity: 0.35 }, 'Training Zone (70%)');
Map.addLayer(ee.Image(0).paint(ee.FeatureCollection([ee.Feature(testZone)]), 1),
             { palette: ['#FF5722'], opacity: 0.35 }, 'Test Zone (30%)');

// Train fused GBT on training zone
var valTrainFused = fused2025.select(OPTICAL_BANDS.concat(SAR_BANDS))
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .addBands(lst2025.rename('LST'));

var valSamplesFused = valTrainFused.sample({
  region: trainZone, scale: PIXEL_SIZE, numPixels: 15000,
  tileScale: 8, seed: 42, geometries: false
}).filter(ee.Filter.notNull(['LST']));

print('Validation training samples:', valSamplesFused.size());

var valRegressorFused = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 300, shrinkage: 0.05, samplingRate: 0.7,
  maxNodes: 8, loss: 'LeastSquares', seed: 42
}).setOutputMode('REGRESSION')
  .train({ features: valSamplesFused, classProperty: 'LST',
           inputProperties: FUSED_LST_BANDS });

// Train optical-only GBT on training zone (ablation)
var valTrainOpt = composite2025.select(OPTICAL_BANDS)
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .addBands(lst2025.rename('LST'));

var valSamplesOpt = valTrainOpt.sample({
  region: trainZone, scale: PIXEL_SIZE, numPixels: 15000,
  tileScale: 8, seed: 42, geometries: false
}).filter(ee.Filter.notNull(['LST']));

var valRegressorOpt = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 300, shrinkage: 0.05, samplingRate: 0.7,
  maxNodes: 8, loss: 'LeastSquares', seed: 42
}).setOutputMode('REGRESSION')
  .train({ features: valSamplesOpt, classProperty: 'LST',
           inputProperties: OPT_LST_BANDS });

// Generate predictions on the test zone
var lstPredFused = fused2025.select(OPTICAL_BANDS.concat(SAR_BANDS))
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .select(FUSED_LST_BANDS)
  .classify(valRegressorFused)
  .rename('LST_pred').clip(roi);

var lstPredOpt = composite2025.select(OPTICAL_BANDS)
  .addBands(composite2025.select(['FVC', 'LSE', 'GHSL_built']))
  .addBands(lc2025.rename('lc'))
  .select(OPT_LST_BANDS)
  .classify(valRegressorOpt)
  .rename('LST_pred_opt').clip(roi);

// Compute RMSE, MAE, Bias against Landsat ST_B10 in the test zone
var lstActual = lst2025.rename('LST_actual');

function computeMetrics(predImage, testGeom) {
  var diff    = lstActual.subtract(predImage.rename('pred')).rename('diff');
  var options = { reducer: ee.Reducer.mean(), geometry: testGeom,
                  scale: PIXEL_SIZE, maxPixels: 1e13, tileScale: 8, bestEffort: true };
  return {
    rmse: ee.Number(diff.pow(2).rename('sq').reduceRegion(
            ee.Reducer.mean(), testGeom, PIXEL_SIZE, null, null, true, 1e13, 8
          ).get('sq')).sqrt(),
    mae:  ee.Number(diff.abs().rename('abs').reduceRegion(
            ee.Reducer.mean(), testGeom, PIXEL_SIZE, null, null, true, 1e13, 8
          ).get('abs')),
    bias: ee.Number(diff.reduceRegion(options).get('diff')),
    diff: diff
  };
}

var metricsFused = computeMetrics(lstPredFused, testZone);
var metricsOpt   = computeMetrics(lstPredOpt,   testZone);

print('--- FUSED GBT VALIDATION METRICS ---');
print('RMSE (°C):', metricsFused.rmse);
print('MAE  (°C):', metricsFused.mae);
print('Bias (°C):', metricsFused.bias);
print('--- OPTICAL-ONLY GBT VALIDATION METRICS ---');
print('RMSE (°C):', metricsOpt.rmse);
print('MAE  (°C):', metricsOpt.mae);
print('Bias (°C):', metricsOpt.bias);

// Outlier-filtered RMSE: exclude pixels outside mean ± 2 SD
// These are structurally unresolvable at 30 m (metal rooftops >44°C,
// deep-shaded water <23°C). Filtering follows Wan et al. (2004).
var lstStats  = lstActual.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
  geometry: testZone, scale: PIXEL_SIZE, maxPixels: 1e13, tileScale: 8, bestEffort: true
});
var lowerBound = ee.Number(lstStats.get('LST_actual_mean'))
                   .subtract(ee.Number(lstStats.get('LST_actual_stdDev')).multiply(2));
var upperBound = ee.Number(lstStats.get('LST_actual_mean'))
                   .add(ee.Number(lstStats.get('LST_actual_stdDev')).multiply(2));

var inlierMask = lstActual.gte(lowerBound).and(lstActual.lte(upperBound));
var filtOpts   = { reducer: ee.Reducer.mean(), geometry: testZone,
                   scale: PIXEL_SIZE, maxPixels: 1e13, tileScale: 8, bestEffort: true };

var rmseFusedFiltered = ee.Number(
  metricsFused.diff.updateMask(inlierMask).pow(2).rename('sq')
    .reduceRegion(filtOpts).get('sq')).sqrt();
var rmseOptFiltered   = ee.Number(
  metricsOpt.diff.updateMask(inlierMask).pow(2).rename('sq')
    .reduceRegion(filtOpts).get('sq')).sqrt();

print('--- OUTLIER-FILTERED RMSE (mean ± 2 SD) ---');
print('Fused GBT filtered RMSE (°C):', rmseFusedFiltered);
print('Optical GBT filtered RMSE (°C):', rmseOptFiltered);
print('Lower bound (°C):', lowerBound);
print('Upper bound (°C):', upperBound);

// Per-class validation (fused GBT)
var perClassVal = ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
  c = ee.Number(c);
  var mask = lc2025.eq(c);
  var actual    = lstActual.updateMask(mask).reduceRegion(
    { reducer: ee.Reducer.mean(), geometry: testZone, scale: PIXEL_SIZE,
      maxPixels: 1e13, tileScale: 8, bestEffort: true }).get('LST_actual');
  var predicted = lstPredFused.updateMask(mask).reduceRegion(
    { reducer: ee.Reducer.mean(), geometry: testZone, scale: PIXEL_SIZE,
      maxPixels: 1e13, tileScale: 8, bestEffort: true }).get('LST_pred');
  return ee.Feature(null, { class_code: c, class: CLASS_NAMES.get(c),
                            actual_LST: actual, predicted_LST: predicted });
}));
print('Per-class mean LST — Actual vs Predicted:', perClassVal);

// Scatter plot (1000 pixel sample)
var scatterData = lstActual.addBands(lstPredFused).sample({
  region: testZone, scale: PIXEL_SIZE, numPixels: 1000, tileScale: 4,
  seed: 99, geometries: false
}).filter(ee.Filter.notNull(['LST_actual', 'LST_pred']));

print(ui.Chart.feature.byFeature(scatterData, 'LST_actual', ['LST_pred'])
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Actual vs Predicted LST — Fused GBT (test zone sample)',
    hAxis: { title: 'Actual LST (°C)' },
    vAxis: { title: 'Predicted LST (°C)' },
    pointSize: 2, colors: ['#1565C0'],
    trendlines: { 0: { type: 'linear', color: 'red', showR2: true,
                       visibleInLegend: true } }
  }));

print('--- END VALIDATION ---');


// ============================================================
//  SECTION 18 — FEATURE IMPORTANCE EXPORTS (FIGURE 4 DATA)
// ============================================================

// LULC RF feature importance
var rfImportance   = classifierFused.explain().get('importance');
var rfImportanceFC = ee.FeatureCollection(
  ee.Dictionary(rfImportance).keys().map(function(b) {
    return ee.Feature(null, { band: b, importance: ee.Dictionary(rfImportance).get(b) });
  })
);
Export.table.toDrive({
  collection: rfImportanceFC,
  description: 'LULC_RF_FeatureImportance',
  folder: EXPORT_FOLDER, fileFormat: 'CSV'
});

// GBT LST feature importance
var gbtImportance   = lstRegressorFused.explain().get('importance');
var gbtImportanceFC = ee.FeatureCollection(
  ee.Dictionary(gbtImportance).keys().map(function(b) {
    return ee.Feature(null, { band: b, importance: ee.Dictionary(gbtImportance).get(b) });
  })
);
Export.table.toDrive({
  collection: gbtImportanceFC,
  description: 'GBT_LST_FeatureImportance',
  folder: EXPORT_FOLDER, fileFormat: 'CSV'
});


// ============================================================
//  SECTION 19 — IMAGE EXPORTS (LULC, LST, SAR)
// ============================================================

var exportRegion = roi;

// LULC maps
Export.image.toDrive({ image: lc2015.toFloat(), description: 'LULC_2015_Fused',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LULC_2015_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: lc2025.toFloat(), description: 'LULC_2025_Fused',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LULC_2025_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: lc2035.toFloat(), description: 'LULC_2035_Projected',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LULC_2035_Projected',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

// LST maps
Export.image.toDrive({ image: lst2015, description: 'LST_2015_Observed',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LST_2015_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: lst2025, description: 'LST_2025_Observed',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LST_2025_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: lst2035, description: 'LST_2035_Projected',
  folder: EXPORT_FOLDER, fileNamePrefix: 'LST_2035_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

// SAR feature stack (6 bands, 30 m)
Export.image.toDrive({ image: sar2025.float(), description: 'SAR_Features_2025',
  folder: EXPORT_FOLDER, fileNamePrefix: 'SAR_Features_2025',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

// SAR false-colour RGB at native 10 m resolution (for Figure 3)
// Rebuilt without unmask(0) and reproject() which cause flat export artefacts
var s1raw = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate(WINDOW_2025.start, WINDOW_2025.end)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH'])
  .map(function(img) {
    return ee.Image(10).multiply(img.log10())
      .rename(['VV_dB', 'VH_dB'])
      .copyProperties(img, img.propertyNames());
  }).median().clip(roi);

var sarFalseColour = s1raw.select('VV_dB').rename('R')
  .addBands(s1raw.select('VH_dB').rename('G'))
  .addBands(s1raw.select('VV_dB').subtract(s1raw.select('VH_dB')).rename('B'))
  .float();

// Print per-band percentile range — use these values for QGIS colour stretch
print('SAR false-colour band stats (p2/p98):', sarFalseColour.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: roi, scale: 30, maxPixels: 1e13, bestEffort: true
}));

Export.image.toDrive({ image: sarFalseColour, description: 'SAR_FalseColour_RGB_2025',
  folder: EXPORT_FOLDER, fileNamePrefix: 'SAR_FalseColour_RGB_2025',
  region: roi, scale: 10, crs: EXPORT_CRS, maxPixels: 1e13 });

// Validation comparison maps
Export.image.toDrive({ image: lst2025.rename('LST_2025_Actual'),
  description: 'LST_2025_Actual', folder: EXPORT_FOLDER,
  fileNamePrefix: 'LST_2025_Actual',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: lstPredFused.rename('LST_2025_Pred_Fused'),
  description: 'LST_2025_Predicted_Fused', folder: EXPORT_FOLDER,
  fileNamePrefix: 'LST_2025_Predicted_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });

Export.image.toDrive({ image: metricsFused.diff.rename('Residual_Fused'),
  description: 'LST_2025_Residual_Fused', folder: EXPORT_FOLDER,
  fileNamePrefix: 'LST_2025_Residual_Fused',
  region: exportRegion, scale: PIXEL_SIZE, crs: EXPORT_CRS, maxPixels: 1e13 });


// ============================================================
//  SECTION 20 — RESULTS SUMMARY CSV EXPORT
// ============================================================

function areaToFC(areaDict, year) {
  return ee.FeatureCollection(ee.List([0, 1, 2]).map(function(c) {
    c = ee.Number(c);
    var areaKm2 = ee.Number(areaDict.get(c.format(), 0)).multiply(0.0009);
    return ee.Feature(null, { metric: 'LULC_Area', year: year,
                              class_code: c, class: CLASS_NAMES.get(c),
                              area_km2: areaKm2 });
  }));
}

// Validation metrics summary
Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(null, { metric: 'RMSE_Fused_full',     value: metricsFused.rmse }),
    ee.Feature(null, { metric: 'RMSE_Fused_filtered', value: rmseFusedFiltered }),
    ee.Feature(null, { metric: 'MAE_Fused',           value: metricsFused.mae }),
    ee.Feature(null, { metric: 'Bias_Fused',          value: metricsFused.bias }),
    ee.Feature(null, { metric: 'RMSE_Optical_full',   value: metricsOpt.rmse }),
    ee.Feature(null, { metric: 'RMSE_Optical_filtered', value: rmseOptFiltered }),
    ee.Feature(null, { metric: 'MAE_Optical',         value: metricsOpt.mae }),
    ee.Feature(null, { metric: 'Bias_Optical',        value: metricsOpt.bias }),
    ee.Feature(null, { metric: 'Outlier_lower_C',     value: lowerBound }),
    ee.Feature(null, { metric: 'Outlier_upper_C',     value: upperBound }),
    ee.Feature(null, { metric: 'OA_Fused_LULC',       value: cmFused.accuracy() }),
    ee.Feature(null, { metric: 'Kappa_Fused_LULC',    value: cmFused.kappa() }),
    ee.Feature(null, { metric: 'OA_Optical_LULC',     value: cmOpt.accuracy() }),
    ee.Feature(null, { metric: 'Kappa_Optical_LULC',  value: cmOpt.kappa() })
  ]).merge(perClassVal),
  description: 'Validation_Metrics_Summary',
  folder: EXPORT_FOLDER, fileFormat: 'CSV'
});

// LULC areas and mean LST across all epochs
var resultsFC = ee.FeatureCollection([
  ee.Feature(null, { metric: 'Mean_LST', year: 2015, value: roiMean(lst2015, 'LST_2015') }),
  ee.Feature(null, { metric: 'Mean_LST', year: 2025, value: roiMean(lst2025, 'LST_2025') }),
  ee.Feature(null, { metric: 'Mean_LST', year: 2035, value: roiMean(lst2035, 'LST_2035') })
]).merge(areaToFC(area2015, 2015))
  .merge(areaToFC(area2025, 2025))
  .merge(areaToFC(area2035, 2035));

Export.table.toDrive({
  collection: resultsFC,
  description: 'LULC_LST_All_Results',
  folder: EXPORT_FOLDER, fileFormat: 'CSV'
});

// ============================================================
//  END OF SCRIPT
// ============================================================
