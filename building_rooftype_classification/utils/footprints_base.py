
import logging
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


import geopandas as gpd

import rasterio
from shapely.geometry import box

from PIL import Image



def extract_footprint_height(dsm_path, footprint_geometry, logger=None):
    """
    Extract height statistics from DSM for a building footprint.
    
    Args:
        dsm_path: Path to the Digital Surface Model (DSM)
        footprint_geometry: Shapely geometry of the building footprint
    
    Returns:
        Dictionary with height statistics, or None if extraction fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        with rasterio.open(dsm_path) as dsm_src:
            
            if dsm_src.crs is None or dsm_src.transform is None:
                logger.error(f"DSM is not georeferenced - cannot extract height for footprint")
                return None
            
            minx, miny, maxx, maxy = footprint_geometry.bounds
            
            # Convert to pixel coordinates
            
            transform = dsm_src.transform
            px1, py1 = ~transform * (minx, maxy)
            px2, py2 = ~transform * (maxx, miny)
            
            row_start = max(0, int(min(py1, py2)))
            row_end = min(dsm_src.height, int(max(py1, py2)) + 1)
            col_start = max(0, int(min(px1, px2)))
            col_end = min(dsm_src.width, int(max(px1, px2)) + 1)
            
            if row_end <= row_start or col_end <= col_start:
                return None
            
            dsm_window = dsm_src.read(1, window=((row_start, row_end), (col_start, col_end)))
            
            if dsm_src.nodata is not None:
                valid_mask = dsm_window != dsm_src.nodata
            else:
                if dsm_src.dtypes[0] == 'uint8':
                    valid_mask = np.ones_like(dsm_window, dtype=bool)
                else:
                    valid_mask = (dsm_window != -9999) & (dsm_window != -32768) & (~np.isnan(dsm_window))
            
            if not np.any(valid_mask):
                return None
                
            dsm_values = dsm_window[valid_mask]
            
            # Calculate statistics
            if len(dsm_values) > 0:
                return {
                    'mean_height': float(np.mean(dsm_values)),
                    'min_height': float(np.min(dsm_values)),
                    'max_height': float(np.max(dsm_values)),
                    'std_height': float(np.std(dsm_values)),
                    'pixel_count': len(dsm_values)
                }
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error extracting height: {e}")
        return None

def extract_footprint_image(orthophoto_path, footprint_geometry, buffer_pixels=10, logger=None):
    """
    Extract image area corresponding to a building footprint from an orthophoto.
    
    Args:
        orthophoto_path: Path to the orthophoto
        footprint_geometry: Shapely geometry of the building footprint
        buffer_pixels: Buffer around footprint in pixels
    
    Returns:
        PIL Image of the extracted footprint area, or None if extraction fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        with rasterio.open(orthophoto_path) as src:

            minx, miny, maxx, maxy = footprint_geometry.bounds
            
            # Convert to pixel coordinates
            transform = src.transform
            px1, py1 = ~transform * (minx, maxy)
            px2, py2 = ~transform * (maxx, miny)
            
            row_start = max(0, int(min(py1, py2)) - buffer_pixels)
            row_end = min(src.height, int(max(py1, py2)) + buffer_pixels)
            col_start = max(0, int(min(px1, px2)) - buffer_pixels)
            col_end = min(src.width, int(max(px1, px2)) + buffer_pixels)
            
            if row_end <= row_start or col_end <= col_start:
                logger.error("Invalid footprint bounds")
                return None
            
            image_data = src.read(window=((row_start, row_end), (col_start, col_end)))
            
            # Handle different number of channels
            if image_data.shape[0] == 1:
                image_array = np.stack([image_data[0]] * 3, axis=0)
            elif image_data.shape[0] >= 3:
                image_array = image_data[:3]
            else:
                logger.error(f"Unsupported number of channels: {image_data.shape[0]}")
                return None
            
            image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Convert to PIL Image
            return Image.fromarray(image_array)
            
    except Exception as e:
        logger.error(f"Error extracting footprint from {os.path.basename(orthophoto_path)}: {e}")
        return None

def classify_footprint_image(model, footprint_image, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES, logger=None):
    """
    Classify a single footprint image using the trained model.
    
    Args:
        model: Trained roof classification model
        footprint_image: PIL Image of the footprint area
    
    Returns:
        Dictionary with classification results
    """
    try:
        # Resize to model input size
        resized_image = footprint_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        
        # Convert to array and normalize
        image_array = img_to_array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
        
    except Exception as e:
        print(f"Error classifying footprint: {e}")
        return None

def process_footprints(model, orthophoto_path, footprint_path, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT,
                       confidence_threshold, dsm_path=None, logger=None):
    """
    Process building footprints by classifying roof types from orthophoto.
    
    Args:
        model: Trained roof classification model
        orthophoto_path: Path to orthophoto image
        footprint_path: Path to footprint shapefile
        confidence_threshold: Minimum confidence for valid classifications
        dsm_path: Optional path to Digital Surface Model for height extraction
    
    Returns:
        GeoDataFrame with classified footprints
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Processing footprints from: {os.path.basename(footprint_path)}")
    logger.info(f"Using orthophoto: {os.path.basename(orthophoto_path)}")
    if dsm_path:
        logger.info(f"Using DSM: {os.path.basename(dsm_path)}")
    
    try:
        footprints_gdf = gpd.read_file(footprint_path)
        logger.info(f"  Loaded {len(footprints_gdf)} footprints")

        with rasterio.open(orthophoto_path) as ortho_src:
            ortho_bounds = box(*ortho_src.bounds)
            ortho_crs = ortho_src.crs

        if ortho_crs is not None and footprints_gdf.crs is not None and footprints_gdf.crs != ortho_crs:
            logger.info(f"  Reprojecting footprints from {footprints_gdf.crs} to {ortho_crs}")
            footprints_gdf = footprints_gdf.to_crs(ortho_crs)

        if not footprints_gdf.empty:
            original_count = len(footprints_gdf)
            footprints_gdf = footprints_gdf[footprints_gdf.geometry.intersects(ortho_bounds)].copy()
            logger.info(f"  Retained {len(footprints_gdf)}/{original_count} footprints overlapping the orthophoto")

        if footprints_gdf.empty:
            logger.warning("  No footprints overlap this orthophoto")
            return footprints_gdf
        
        footprints_gdf['roof_class'] = 'unknown'
        footprints_gdf['confidence'] = 0.0
        footprints_gdf['classified'] = False
        
        for class_name in CLASS_NAMES:
            footprints_gdf[f'prob_{class_name[:8]}'] = 0.0
        
        if dsm_path:
            footprints_gdf['mean_height'] = np.nan
            footprints_gdf['min_height'] = np.nan
            footprints_gdf['max_height'] = np.nan
            footprints_gdf['std_height'] = np.nan
            footprints_gdf['height_px'] = 0
        
        classified_count = 0
        height_extracted_count = 0
        
        for idx, footprint in footprints_gdf.iterrows():
            try:
                logger.debug(f"Processing footprint {idx+1}/{len(footprints_gdf)}")
                
                if dsm_path:
                    height_stats = extract_footprint_height(dsm_path, footprint.geometry)
                    if height_stats:
                        footprints_gdf.at[idx, 'mean_height'] = height_stats['mean_height']
                        footprints_gdf.at[idx, 'min_height'] = height_stats['min_height']
                        footprints_gdf.at[idx, 'max_height'] = height_stats['max_height']
                        footprints_gdf.at[idx, 'std_height'] = height_stats['std_height']
                        footprints_gdf.at[idx, 'height_px'] = height_stats['pixel_count']
                        height_extracted_count += 1
                
                footprint_image = extract_footprint_image(
                    orthophoto_path, 
                    footprint.geometry,
                    buffer_pixels=5
                )
                
                if footprint_image is None:
                    continue
                
                classification = classify_footprint_image(model, footprint_image, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, CLASS_NAMES=CLASS_NAMES, logger=logging.getLogger(__name__))
                
                if classification is None:
                    continue
                
                if classification['confidence'] >= confidence_threshold:
                    footprints_gdf.at[idx, 'roof_class'] = classification['predicted_class']
                    footprints_gdf.at[idx, 'confidence'] = classification['confidence']
                    footprints_gdf.at[idx, 'classified'] = True
                    
                    for class_name in CLASS_NAMES:
                        footprints_gdf.at[idx, f'prob_{class_name[:8]}'] = classification['class_probabilities'][class_name]
                    
                    classified_count += 1
                    
                    if classified_count % 10 == 0:
                        logger.info(f"    Classified {classified_count} footprints...")
                
            except Exception as e:
                logger.error(f"Error processing footprint {idx}: {e}")
                continue
        
        logger.info(f"  Successfully classified {classified_count}/{len(footprints_gdf)} footprints")
        if dsm_path:
            logger.info(f"  Successfully extracted height for {height_extracted_count}/{len(footprints_gdf)} footprints")
        
        return footprints_gdf
        
    except Exception as e:
        logger.error(f"Error processing footprints: {e}")
        return None


def pixel_to_geographic(x, y, transform):
    """Convert pixel coordinates to geographic coordinates using rasterio transform."""
    if transform is None:
        return x, y
    
    geo_x, geo_y = transform * (x, y)
    return geo_x, geo_y



def print_footprint_summary(all_results, CLASS_NAMES):
    """Print summary statistics for footprint classifications."""
    if not all_results:
        print("\n" + "="*60)
        print("NO CLASSIFICATIONS COMPLETED")
        print("="*60)
        return
    
    total_footprints = len(all_results)
    
    print("\n" + "="*60)
    print("FOOTPRINT CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total classified footprints: {total_footprints}")
    
    # Class distribution
    class_counts = {}
    confidences = []
    heights = []
    
    for result in all_results:
        roof_class = result['roof_class']
        confidence = result['confidence']
        
        class_counts[roof_class] = class_counts.get(roof_class, 0) + 1
        confidences.append(confidence)
        
        # Collect height data if available
        if 'mean_height' in result and not pd.isna(result['mean_height']):
            heights.append(result['mean_height'])
    
    print(f"\nClass Distribution:")
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / total_footprints) * 100 if total_footprints > 0 else 0
        print(f"  {class_name:12}: {count:4d} ({percentage:5.1f}%)")
    
    if confidences:
        confidences = np.array(confidences)
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std:  {np.std(confidences):.3f}")
        print(f"  Min:  {np.min(confidences):.3f}")
        print(f"  Max:  {np.max(confidences):.3f}")
    
    if heights:
        heights = np.array(heights)
        print(f"\nHeight Statistics (from DSM - pixel values, not meters):")
        print(f"  Buildings with height data: {len(heights)}/{total_footprints}")
        print(f"  Mean pixel value: {np.mean(heights):.2f}")
        print(f"  Std pixel value:  {np.std(heights):.2f}")
        print(f"  Min pixel value:  {np.min(heights):.2f}")
        print(f"  Max pixel value:  {np.max(heights):.2f}")
        print(f"  Note: These are pixel values (0-255), not actual heights in meters")


