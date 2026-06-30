import os
import sys
import logging

from tensorflow.keras.models import load_model


def find_best_model():
    """Automatically find the best available model in the models directory."""
    models_dir = "models"
    model_candidates = [
        "fine_tuned_vgg16_final.keras",
        "best_fine_tuned_vgg16.keras",
        "fine_tuned_vgg16_final.h5",
        "best_fine_tuned_vgg16.h5",
        "fine_tuned_vgg16_final.wheight.h5",
        "fine_tuned_vgg16_final_savedmodel"
    ]
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found!")
    
    for model_name in model_candidates:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return model_path
    
    raise FileNotFoundError(
        f"No trained model found in '{models_dir}' directory! Please ensure you have one of the following files:\n" +
        "\n".join([f"  - {models_dir}/{path}" for path in model_candidates])
    )

def load_trained_model(model_path=None, logger=None):
    
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("Load the trained model.")
    if model_path is None:
        model_path = find_best_model()
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = load_model(model_path)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def get_orthophoto_footprint_dsm_triplets(input_dir, SUPPORTED_FORMATS, dsm_dir=None, footprints_dir=None, logger=None):
    """
    Get triplets of orthophotos, their corresponding footprint shapefiles, and DSM files.
    
    Args:
        input_dir: Directory containing orthophotos
        footprints_dir: Directory containing footprint shapefiles (if None, uses input_dir/footprints)
        dsm_dir: Optional directory containing DSM files (if None, looks for DSM files in input_dir)
    
    Returns:
        List of tuples: (orthophoto_path, footprint_path, dsm_path or None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Use custom footprints directory if provided, otherwise default to input_dir/footprints
    if footprints_dir is None:
        footprints_dir = os.path.join(input_dir, 'footprints')
    
    if not os.path.exists(footprints_dir):
        raise FileNotFoundError(f"Footprints directory not found: {footprints_dir}")
    
    logger.info(f"Using footprints directory: {footprints_dir}")

    shared_shapefiles = sorted(
        os.path.join(footprints_dir, filename)
        for filename in os.listdir(footprints_dir)
        if filename.lower().endswith('.shp')
    )
    
    if dsm_dir is None:
        dsm_dir = input_dir
    elif not os.path.exists(dsm_dir):
        logger.warning(f"DSM directory not found: {dsm_dir}, skipping DSM processing")
        dsm_dir = None
    
    triplets = []
    
    # Get all orthophoto files
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            orthophoto_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            possible_footprint_names = [
                f"{base_name}_footprints.shp",
                f"{base_name}.shp", 
                f"{base_name}_buildings.shp",
                f"{base_name}_emprise.shp"
            ]
            
            footprint_path = None
            for footprint_name in possible_footprint_names:
                potential_path = os.path.join(footprints_dir, footprint_name)
                if os.path.exists(potential_path):
                    footprint_path = potential_path
                    break

            if footprint_path is None and len(shared_shapefiles) == 1:
                footprint_path = shared_shapefiles[0]
                logger.info(
                    f"Using shared footprint shapefile for {filename}: "
                    f"{os.path.basename(footprint_path)}"
                )
            
            dsm_path = None
            if dsm_dir:
                possible_dsm_names = [
                    f"{base_name}_dsm.tif",
                    f"{base_name}_DSM.tif",
                    f"{base_name}.tif",
                    f"{base_name}_height.tif",
                    f"{base_name}_heights.tif"
                ]
                
                for dsm_name in possible_dsm_names:
                    potential_dsm_path = os.path.join(dsm_dir, dsm_name)
                    if os.path.exists(potential_dsm_path):
                        dsm_path = potential_dsm_path
                        break
            
            if footprint_path:
                triplets.append((orthophoto_path, footprint_path, dsm_path))
                dsm_status = f" + DSM: {os.path.basename(dsm_path)}" if dsm_path else " (no DSM)"
                logger.info(f"Found: {filename} -> {os.path.basename(footprint_path)}{dsm_status}")
            else:
                logger.info(f"No footprint shapefile found for {filename}")
    
    if not triplets:
        raise ValueError(f"No orthophoto-footprint pairs found in {input_dir}")
    
    return triplets

def save_classified_footprints(footprints_gdf, output_path, logger=None):
    """Save classified footprints to shapefile."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if footprints_gdf is None or len(footprints_gdf) == 0:
        logger.error("No footprints to save")
        return
    
    try:
        footprints_gdf.to_file(output_path, driver='ESRI Shapefile')
        logger.info(f"Classified footprints saved: {output_path}")
        
        geojson_path = output_path.replace('.shp', '.geojson')
        footprints_gdf.to_file(geojson_path, driver='GeoJSON')
        logger.info(f"GeoJSON saved: {geojson_path}")
        
        classified_footprints = footprints_gdf[footprints_gdf['classified'] == True]
        if len(classified_footprints) > 0:
            logger.info(f"Classification summary:")
            class_counts = classified_footprints['roof_class'].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(classified_footprints)) * 100
                logger.info(f"    {class_name}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error saving classified footprints: {e}")
