"""
Building Roof Type Classification - Footprint-Based Inference

Classifies roof types for building footprints using orthophotos and a trained model.
Optionally extracts height statistics from DSM files.

Workflow:
1. Load building footprints and orthophotos
2. Extract roof areas for each footprint
3. Classify roof type and confidence
4. Optionally extract height stats from DSM
5. Save labeled footprints

Features:
- Supports RGB, RGBA, RGBI, multi-channel, and grayscale images
- Handles georeferenced data
- Batch processing and confidence scoring
- Optional DSM-based height extraction

Usage:
    python orthophoto_inference.py --input_dir <orthophotos_dir> [--dsm_dir <dsm_dir>] [--confidence_threshold <float>] [--visualize] [--output_csv <csv_file>]

Requirements:
    pip install opencv-python geopandas shapely fiona rasterio
"""

import os
import sys
import logging
import argparse

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0=all, 1=info, 2=warning, 3=error
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('TF_GPU_THREAD_MODE', 'gpu_private')
os.environ.setdefault('TF_GPU_THREAD_COUNT', '1')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.loader import *
from utils.saver import *
from utils.visualiser import *
from utils.footprints import *


import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import cv2
from skimage.util import view_as_windows
from PIL import Image, ImageDraw, ImageFont

import geopandas as gpd
from shapely.geometry import Polygon, box
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_bounds as transform_from_bounds



# Configuration
IMG_HEIGHT = 140
IMG_WIDTH = 140
CLASS_NAMES = ['complex', 'flat', 'gable', 'hip', 'L-shaped', 'pyramid']
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2', '.webp', '.gif'}

CLASS_COLORS = {
    'complex': (0, 255, 255),   
    'flat': (255, 0, 0),        
    'gable': (0, 255, 0),       
    'hip': (128, 0, 128),       
    'L-shaped': (255, 192, 203),
    'pyramid': (255, 0, 255)    
}



def main():
    parser = argparse.ArgumentParser(
        description="Building Roof Type Classification - Footprint-Based Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic footprint classification (requires footprints/ subdirectory)
  python orthophoto_inference.py --input_dir test_orthophotos/
  
  # With DSM for height extraction
  python orthophoto_inference.py --input_dir test_orthophotos/ --dsm_dir test_dsm/
  
  # With visualization of classified footprints
  python orthophoto_inference.py --input_dir test_orthophotos/ --visualize

  
Directory Structure:
  test_orthophotos/
  ├── orthophoto1.tif
  ├── orthophoto2.tif
  └── footprints/
      ├── orthophoto1_footprints.shp  (or orthophoto1.shp)
      └── orthophoto2_footprints.shp  (or orthophoto2.shp)
  
  Optional DSM Structure:
  test_dsm/
  ├── orthophoto1_dsm.tif  (or orthophoto1_DSM.tif)
  └── orthophoto2_dsm.tif
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing orthophotos and footprints/ subdirectory')
    parser.add_argument('--footprints_dir', default=None,
                       help='Directory containing footprint shapefiles (if not specified, looks for input_dir/footprints)')
    parser.add_argument('--dsm_dir', default=None,
                       help='Directory containing DSM files for height extraction (optional)')
    parser.add_argument('--model_path', default=None,
                       help='Path to trained model (auto-detect if not specified)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for classifications (default: 0.5)')
    parser.add_argument('--output_csv', default=None,
                       help='Output CSV file for classification results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of classified footprints')
    parser.add_argument('--output_dir', default='orthophoto_results',
                       help='Output directory for results (default: orthophoto_results)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FOOTPRINT-BASED ROOF CLASSIFICATION")
    logger.info("="*60)
    logger.info(f"Confidence threshold: {args.confidence_threshold}")
    if args.dsm_dir:
        logger.info(f"DSM directory: {args.dsm_dir}")
    
    
    # Load model
    model = load_trained_model(args.model_path)
    
    # Get orthophoto-footprint-DSM triplets
    logger.info(f"\nScanning directory: {args.input_dir}")
    if args.footprints_dir:
        logger.info(f"Using custom footprints directory: {args.footprints_dir}")
    if args.dsm_dir:
        logger.info(f"DSM directory: {args.dsm_dir}")
    try:
        triplets = get_orthophoto_footprint_dsm_triplets(
            args.input_dir, SUPPORTED_FORMATS, 
            dsm_dir=args.dsm_dir, 
            footprints_dir=args.footprints_dir
        )
        logger.info(f"Found {len(triplets)} orthophoto-footprint pairs")
        dsm_count = sum(1 for _, _, dsm in triplets if dsm is not None)
        if dsm_count > 0:
            logger.info(f"DSM files available for {dsm_count} orthophotos")
    except Exception as e:
        logger.error(f"Error finding orthophoto-footprint pairs: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each orthophoto-footprint-DSM triplet
    all_results = []
    
    for i, (orthophoto_path, footprint_path, dsm_path) in enumerate(triplets, 1):
        logger.info(f"\n[{i}/{len(triplets)}] " + "="*50)
        
        classified_footprints = process_footprints(
            model, orthophoto_path, footprint_path, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH,
            confidence_threshold=args.confidence_threshold,
            dsm_path=dsm_path,
            logger=logger
        )
        
        if classified_footprints is not None:
            base_name = os.path.splitext(os.path.basename(orthophoto_path))[0]
            output_shapefile = os.path.join(args.output_dir, f"{base_name}_classified.shp")
            save_classified_footprints(classified_footprints, output_shapefile)
            
            if args.visualize:
                viz_filename = f"{base_name}_classified_footprints.png"
                viz_path = os.path.join(args.output_dir, viz_filename)
                create_footprint_visualization(orthophoto_path, classified_footprints, viz_path)
            
            for idx, footprint in classified_footprints.iterrows():
                if footprint['classified']:
                    result = {
                        'orthophoto': os.path.basename(orthophoto_path),
                        'footprint_id': idx,
                        'roof_class': footprint['roof_class'],
                        'confidence': footprint['confidence']
                    }
                    
                    if dsm_path and not pd.isna(footprint.get('mean_height', np.nan)):
                        result['mean_height'] = footprint['mean_height']
                        result['min_height'] = footprint['min_height']
                        result['max_height'] = footprint['max_height']
                        result['std_height'] = footprint['std_height']
                        result['height_pixels'] = footprint['height_px']
                    
                    for class_name in CLASS_NAMES:
                        result[f'prob_{class_name[:8]}'] = footprint[f'prob_{class_name[:8]}']
                    
                    all_results.append(result)

    if args.output_csv and all_results:
        csv_path = args.output_csv if os.path.dirname(args.output_csv) else os.path.join(args.output_dir, args.output_csv)
        save_footprint_results_to_csv(all_results, csv_path)
    
    print_footprint_summary(all_results, CLASS_NAMES)
    
    logger.info(f"\n" + "="*60)
    logger.info("PROCESSING COMPLETED")
    logger.info("="*60)
    
    logger.info("\nFiles generated:")
    logger.info(f"Classified shapefiles: {args.output_dir}/*_classified.shp")
    logger.info(f"GeoJSON files: {args.output_dir}/*_classified.geojson")
    if args.output_csv:
        logger.info(f"CSV results: {csv_path}")
    if args.visualize:
        logger.info(f"Footprint visualizations: {args.output_dir}/*_classified_footprints.png")
    if args.dsm_dir:
        logger.info(f"Height statistics included in outputs")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    main()
