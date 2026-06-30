#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import cv2
import utils as utils
import time
import matplotlib.pyplot as plt
import glob
import os
import argparse
import sys
import logging
import subprocess
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# Argument parser
parser = argparse.ArgumentParser(description="DSMNet - DSM generation from orthophotos")
parser.add_argument("--input_path", default='/workspace/data/input/ortho', 
                    help="Path to the orthophotos directory (default:/workspace/data/input/ortho)")
parser.add_argument("--output_path", default='./output/Vaihingen',
                    help="Output directory for DSMs (default: ./output/Vaihingen)")
parser.add_argument("--debug", action="store_true",
                    help="Enable debugging mode with detailed logging")
parser.add_argument("--refinement_iterations", type=int, default=1,
                    help="Number of refinement iterations (default: 1)")
parser.add_argument("--step_size_factor", type=float, default=0.5,
                    help="Factor for sliding window step size (default: 0.5, smaller = more overlap)")
parser.add_argument("--disable_refinement", action="store_true",
                    help="Disable refinement process entirely")
parser.add_argument("--batch_size", type=int, default=0,
                    help="Inference batch size for crop processing (default: 0 = auto)")
parser.add_argument(
    "--dataset_name",
    default=os.environ.get("CITYZEN_DSMNET_DATASET", "Bologna"),
    help="DSMNet inference profile: Vaihingen, DFC2018, or Bologna",
)
parser.add_argument(
    "--checkpoint_dir",
    default=os.environ.get("CITYZEN_DSMNET_CHECKPOINT_DIR", "./checkpoints/Bologna"),
    help="Directory containing mtl.weights.h5 and refinement.weights.h5",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=None,
    help="Override semantic output classes (optional)",
)
parser.add_argument(
    "--building_class_index",
    type=int,
    default=int(os.environ.get("CITYZEN_DSMNET_BUILDING_CLASS_INDEX", "1")),
    help="Semantic class index used for extracting building footprints",
)
parser.add_argument(
    "--dsm_clamp_min",
    type=float,
    default=float(
        os.environ.get(
            "CITYZEN_DSM_CLAMP_MIN",
            os.environ.get("CITYZEN_NDSM_CLAMP_MIN", "0.0"),
        )
    ),
    help="Clamp saved DSM values below this threshold (default: 0.0 for nDSM-style output)",
)

args = parser.parse_args()
if args.num_classes is None and os.environ.get("CITYZEN_DSMNET_NUM_CLASSES"):
    args.num_classes = int(os.environ["CITYZEN_DSMNET_NUM_CLASSES"])

# Configuration des paramètres de raffinement
correction = not args.disable_refinement
refinement_iterations = args.refinement_iterations
step_size_factor = args.step_size_factor

# Configure logging based on debug flag
if args.debug:
    log_level = "DEBUG"
else:
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=getattr(logging, log_level), 
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout) 
    ]
)
# 
dsm_logger = logging.getLogger(__name__)
dsm_logger.info(f"DSMNet: Initialisation with log level:{log_level}")
dsm_logger.info(f"Input path: {args.input_path}")
dsm_logger.info(f"Output path: {args.output_path}")
dsm_logger.info(f"Refinement enabled: {correction}")
if correction:
    dsm_logger.info(f"Refinement iterations: {refinement_iterations}")
    dsm_logger.info(f"Step size factor: {step_size_factor}")
else:
    dsm_logger.info("Refinement process disabled")

import tensorflow as tf
from skimage import measure
from shapely.geometry import shape, Polygon
import geopandas as gpd
import rasterio
from rasterio.features import shapes

# Check for GPU(s)
dsm_logger.info("Configuring TensorFlow to use the GPU...")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    dsm_logger.info(f"GPU(s) detected: {len(gpus)}")
    
    if gpus:
        for gpu in gpus:
            dsm_logger.info(f"  - {gpu}")
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        dsm_logger.info("TensorFlow configured to use the GPU")
        
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            dsm_logger.info(f"GPU test passed: {test_tensor.device}")
            
    else:
        dsm_logger.warning("No GPU detected, CPU usage")
        
except Exception as e:
    dsm_logger.error(f"Error during GPU configuration: {e}")
    dsm_logger.warning("Default CPU usage")

gpu_available = bool(gpus) if 'gpus' in locals() else False

from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.densenet import DenseNet121
from PIL import Image

from nets import Autoencoder, MTL
from utils import *
from tifffile import *
from tqdm import tqdm

import sys
import logging
import rasterio
from rasterio.transform import from_bounds

DATASET_PROFILES = {
    "Vaihingen": {
        "num_classes": 6,
        "building_class_index": 1,
        "color_map": np.array(
            [
                [255, 255, 255],
                [0, 0, 255],
                [0, 255, 255],
                [0, 255, 0],
                [255, 255, 0],
                [255, 0, 0],
            ],
            dtype=np.uint8,
        ),
    },
    "DFC2018": {
        "num_classes": 20,
        "building_class_index": 1,
        "color_map": None,
    },
    "Bologna": {
        "num_classes": 2,
        "building_class_index": 1,
        "color_map": np.array(
            [
                [255, 255, 255],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        ),
    },
}


def resolve_dataset_name(requested_dataset_name, checkpoint_dir):
    if requested_dataset_name and requested_dataset_name != "auto":
        return requested_dataset_name

    if checkpoint_dir:
        checkpoint_dir_name = Path(checkpoint_dir).name.lower()
        for profile_name in DATASET_PROFILES:
            if profile_name.lower() in checkpoint_dir_name:
                return profile_name

    for profile_name in ("Bologna", "Vaihingen", "DFC2018"):
        if Path("./checkpoints") .joinpath(profile_name, "mtl.weights.h5").exists():
            return profile_name

    return "Vaihingen"


def resolve_color_map(dataset_name, num_classes):
    profile = DATASET_PROFILES.get(dataset_name, {})
    if profile.get("color_map") is not None and int(profile["num_classes"]) == int(num_classes):
        return profile["color_map"]

    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def extract_building_footprints(semantic_mask, transform, crs, building_class=1):
    """
    Extract building footprints from semantic segmentation mask
    
    Args:
        semantic_mask: 2D numpy array with class predictions
        transform: rasterio transform object
        crs: coordinate reference system
        building_class: class index used to represent buildings
    
    Returns:
        GeoDataFrame with building footprints
    """
    try:
        # Create binary mask for buildings
        building_mask = (semantic_mask == building_class).astype(np.uint8)
        
        building_shapes = []
        for geom, value in shapes(building_mask, mask=building_mask, transform=transform):
            if value == 1:
                building_shapes.append(shape(geom))
        
        if building_shapes:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {'bldg_id': range(len(building_shapes))}, 
                geometry=building_shapes, 
                crs=crs
            )
            
            gdf['area'] = gdf.geometry.area
            gdf = gdf[gdf.area >= 10.0]
            
            gdf.geometry = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
            
            return gdf
        else:
            return gpd.GeoDataFrame(columns=['bldg_id', 'area'], crs=crs)
            
    except Exception as e:
        dsm_logger.error(f"Error in extract_building_footprints: {str(e)}")
        return gpd.GeoDataFrame(columns=['bldg_id', 'area'], crs=crs)

datasetName = resolve_dataset_name(args.dataset_name, args.checkpoint_dir)
dataset_profile = DATASET_PROFILES.get(datasetName, DATASET_PROFILES["Vaihingen"])
resolved_num_classes = int(args.num_classes or dataset_profile["num_classes"])
resolved_building_class_idx = int(args.building_class_index)
resolved_checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("./checkpoints") / datasetName

if not resolved_checkpoint_dir.exists():
    dsm_logger.error(f"Checkpoint directory not found: {resolved_checkpoint_dir}")
    sys.exit(1)

sem_flag = True
cropSize=320
num_classes = resolved_num_classes

def resolve_batch_size(requested_batch_size, use_gpu, refinement_enabled):
  if requested_batch_size > 0:
    return requested_batch_size
  if use_gpu:
    return 4 if refinement_enabled else 8
  return 1

effective_batch_size = resolve_batch_size(args.batch_size, gpu_available, correction)

predCheckPointPath = str(resolved_checkpoint_dir / 'mtl.weights.h5')
corrCheckPointPath = str(resolved_checkpoint_dir / 'refinement.weights.h5')

dsm_logger.info(f"Resolved DSMNet dataset profile: {datasetName}")
dsm_logger.info(f"Resolved number of semantic classes: {num_classes}")
dsm_logger.info(f"Resolved checkpoint directory: {resolved_checkpoint_dir}")
dsm_logger.info(f"Resolved building class index: {resolved_building_class_idx}")

if not Path(predCheckPointPath).exists():
    dsm_logger.error(f"MTL checkpoint not found: {predCheckPointPath}")
    sys.exit(1)

backboneNet=DenseNet121(weights=None, include_top=False, input_tensor=Input(shape=(cropSize,cropSize,3)))

net = MTL(backboneNet, num_classes=num_classes)
random_input = np.zeros((1, cropSize, cropSize, 3), dtype=np.float32)
net(random_input, training=False)


dsm_logger.info(f"Loading model weights from {predCheckPointPath}")
net.load_weights(predCheckPointPath)
dsm_logger.debug("Model weights loaded successfully")

if(correction):
  if not Path(corrCheckPointPath).exists():
    dsm_logger.warning(
      f"Refinement checkpoint not found at {corrCheckPointPath}; DSM refinement will be disabled"
    )
    correction = False

if(correction):
  autoencoder = Autoencoder()
  correction_input_channels = 1 + 3 + num_classes + 3
  sample_input = np.zeros((1, cropSize, cropSize, correction_input_channels), dtype=np.float32)
  autoencoder(sample_input, training=False)
  autoencoder.load_weights(corrCheckPointPath)

@tf.function(
  input_signature=[tf.TensorSpec(shape=(None, cropSize, cropSize, 3), dtype=tf.float32)],
  reduce_retracing=True
)
def infer_batch(batch_rgb):
  dsm_output, sem_output, norm_output = net(batch_rgb, training=False)
  if correction:
    refined_dsm = dsm_output
    for iteration in range(refinement_iterations):
      correctionInput = tf.concat([refined_dsm, norm_output, sem_output, batch_rgb], axis=-1)
      noise = autoencoder(correctionInput, training=False)
      refined_dsm = refined_dsm - noise * tf.cast(0.8 ** iteration, refined_dsm.dtype)
    dsm_output = refined_dsm
  return dsm_output, sem_output, norm_output

infer_batch(tf.zeros((1, cropSize, cropSize, 3), dtype=tf.float32))
dsm_logger.info(f"Crop inference batch size: {effective_batch_size}")

tile_mse   = 0
total_mse  = 0

tile_rmse  = 0
total_rmse = 0

tile_mae   = 0
total_mae  = 0

tile_time  = 0
total_time = 0

target_dir = args.input_path
tif_files = sorted(glob.glob(os.path.join(target_dir, '*.tif')))

output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)

tilesLen = len(tif_files)

if tilesLen == 0:
  dsm_logger.error(f"No TIF files found in {target_dir}")
  sys.exit(1)

dsm_logger.info(f"Found {tilesLen} TIF files to process")

for tif_path in tqdm(tif_files, desc="Processing TIF files"):
  dsm_logger.info(f"Processing {tif_path}")
  
  with rasterio.open(tif_path) as src:
    rgb_data = src.read()
    transform = src.transform
    crs = src.crs
    dsm_logger.debug(f"Read image shape: {rgb_data.shape}, CRS: {crs}, Transform: {transform}")

    if rgb_data.shape[0] <= 4:
      rgb_tile = np.transpose(rgb_data, (1, 2, 0))
      dsm_logger.debug(f"Transposed rgb_data to shape: {rgb_tile.shape}")
    else:
      rgb_tile = rgb_data
      dsm_logger.debug(f"rgb_tile shape (no transpose): {rgb_tile.shape}")

    if rgb_tile.shape[2] == 4:
      rgb_tile = rgb_tile[:, :, :3]
      dsm_logger.debug("Removed alpha channel from rgb_tile")
    elif rgb_tile.shape[2] == 1:
      rgb_tile = np.repeat(rgb_tile, 3, axis=2)
      dsm_logger.debug("Repeated single channel to 3 channels in rgb_tile")

    rgb_tile = rgb_tile.astype(np.float32) / 255.0
    dsm_logger.debug(f"Normalized rgb_tile to float32 in range [0,1]")

  original_h, original_w = rgb_tile.shape[:2]
  h, w = original_h, original_w
  dsm_logger.debug(f"Tile size after normalization: {h}x{w}")
  if h < cropSize or w < cropSize:
    pad_h = max(0, cropSize - h)
    pad_w = max(0, cropSize - w)
    rgb_tile = np.pad(rgb_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h, w = rgb_tile.shape[:2]
    dsm_logger.debug(f"Padded rgb_tile to size: {h}x{w}")

  coordinates = []
  step_size = max(1, int(cropSize * step_size_factor))
  for x1, x2, y1, y2 in sliding_window(rgb_tile, step=step_size, window_size=(cropSize, cropSize)):
    coordinates.append((y1, y2, x1, x2))
  dsm_logger.info(
      f"Generated {len(coordinates)} crops for {os.path.basename(tif_path)} "
      f"with step size {step_size} and batch size {effective_batch_size}"
  )

  prob_matrix = gaussian_kernel(cropSize, cropSize)
  prob_matrix_expanded = prob_matrix[..., np.newaxis]

  pred_sum = np.zeros((h, w), dtype=np.float32)
  weight_sum = np.zeros((h, w), dtype=np.float32)
  sem_sum = np.zeros((h, w, num_classes), dtype=np.float32)

  total_batches = (len(coordinates) + effective_batch_size - 1) // effective_batch_size
  for batch_index, start_idx in enumerate(
      tqdm(
          range(0, len(coordinates), effective_batch_size),
          total=total_batches,
          desc=f"Processing crops for {os.path.basename(tif_path)}",
          leave=False
      )
  ):
    batch_coords = coordinates[start_idx:start_idx + effective_batch_size]
    batch_rgb = np.stack(
        [rgb_tile[y1:y2, x1:x2, :] for y1, y2, x1, x2 in batch_coords],
        axis=0
    ).astype(np.float32, copy=False)

    if batch_index == 0:
      try:
        gpu_usage = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        dsm_logger.info(f"GPU usage before inference: {gpu_usage.strip()}")
      except Exception as e:
        dsm_logger.debug(f"Impossible to check GPU usage: {e}")

    dsm_output, sem_output, norm_output = infer_batch(batch_rgb)
    dsm_logger.debug(
        f"Batch {batch_index}: outputs shapes dsm={dsm_output.shape}, "
        f"sem={sem_output.shape}, norm={norm_output.shape}"
    )

    if batch_index == 0:
      try:
        gpu_usage = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        dsm_logger.info(f"GPU use after inference: {gpu_usage.strip()}")
      except Exception as e:
        dsm_logger.debug(f"Unable to check GPU usage: {e}")

    dsm_output_np = np.squeeze(dsm_output.numpy(), axis=-1).astype(np.float32, copy=False)
    sem_output_np = sem_output.numpy().astype(np.float32, copy=False)
    weighted_dsm = dsm_output_np * prob_matrix[np.newaxis, :, :]
    weighted_sem = sem_output_np * prob_matrix_expanded[np.newaxis, :, :, :]

    for batch_item_index, (y1, y2, x1, x2) in enumerate(batch_coords):
      pred_sum[y1:y2, x1:x2] += weighted_dsm[batch_item_index]
      weight_sum[y1:y2, x1:x2] += prob_matrix
      sem_sum[y1:y2, x1:x2, :] += weighted_sem[batch_item_index]

  pred = np.divide(
      pred_sum,
      weight_sum,
      out=np.zeros_like(pred_sum),
      where=weight_sum > 0
  )
  pred = pred[:original_h, :original_w]

  valid_pred_mask = np.isfinite(pred)
  clamped_low_count = int(np.count_nonzero(valid_pred_mask & (pred < np.float32(args.dsm_clamp_min))))
  if clamped_low_count:
    pred = pred.copy()
    pred[valid_pred_mask] = np.maximum(pred[valid_pred_mask], np.float32(args.dsm_clamp_min))
    dsm_logger.info(
        "Clamped %d DSM pixels below %.3f for %s",
        clamped_low_count,
        float(args.dsm_clamp_min),
        os.path.basename(tif_path),
    )
  dsm_logger.debug(f"Final DSM prediction shape: {pred.shape}")
  
  # Process semantic segmentation
  sem_final = np.divide(
      sem_sum,
      weight_sum[..., np.newaxis],
      out=np.zeros_like(sem_sum),
      where=weight_sum[..., np.newaxis] > 0
  )
  sem_final = sem_final[:original_h, :original_w, :]
  
  # Get building class (blue color in semantic_rgb corresponds to buildings)
  building_class_idx = resolved_building_class_idx

  # Create building footprint mask
  building_footprint = np.argmax(sem_final, axis=2) == building_class_idx
  dsm_logger.debug(f"Final semantic prediction shape: {sem_final.shape}")
  dsm_logger.debug(f"Building footprint shape: {building_footprint.shape}")

  filename = os.path.splitext(os.path.basename(tif_path))[0]
  output_tif_path = os.path.join(output_dir, filename + '_dsm.tif')
  building_output_path = os.path.join(output_dir, filename + '_buildings.tif')
  semantic_output_path = os.path.join(output_dir, filename + '_semantic_rgb.tif')

  # Save DSM with georeference
  with rasterio.open(
    output_tif_path,
    'w',
    driver='GTiff',
    height=pred.shape[0],
    width=pred.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=crs,
    transform=transform,
    compress='lzw',
    nodata=-9999) as dst:
    dst.write(pred.astype(np.float32), 1)
    dsm_logger.debug(f"Written DSM prediction to {output_tif_path}")

  # Save building footprints
  with rasterio.open(
    building_output_path,
    'w',
    driver='GTiff',
    height=building_footprint.shape[0],
    width=building_footprint.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    compress='lzw') as dst:
    dst.write(building_footprint.astype(np.uint8), 1)
    dsm_logger.debug(f"Written building footprints to {building_output_path}")

  # Save semantic RGB visualization
  color_map = resolve_color_map(datasetName, num_classes)
  
  class_map = np.argmax(sem_final, axis=2)
  semantic_rgb = color_map[class_map]
  
  with rasterio.open(
    semantic_output_path,
    'w',
    driver='GTiff',
    height=semantic_rgb.shape[0],
    width=semantic_rgb.shape[1],
    count=3,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    compress='lzw') as dst:
    for i in range(3):
      dst.write(semantic_rgb[:, :, i].astype(np.uint8), i+1)
    dsm_logger.debug(f"Written semantic RGB to {semantic_output_path}")

  dsm_logger.info(f"Saved DSM: {output_tif_path}")
  dsm_logger.info(f"Saved buildings: {building_output_path}")  
  dsm_logger.info(f"Saved semantic RGB: {semantic_output_path}")

  # Save semantic segmentation
  sem_class = np.argmax(sem_final, axis=2).astype(np.uint8)
  sem_tif_path = os.path.join(output_dir, filename + '_semantic.tif')
  with rasterio.open(
    sem_tif_path,
    'w',
    driver='GTiff',
    height=sem_class.shape[0],
    width=sem_class.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    compress='lzw',
    nodata=255) as dst:
    dst.write(sem_class, 1)
    dsm_logger.debug(f"Written semantic prediction to {sem_tif_path}")

  # Extract and save building footprints
  try:
    footprints_gdf = extract_building_footprints(sem_class, transform, crs, building_class_idx)
    
    if len(footprints_gdf) > 0:
      footprints_dir = os.path.join(output_dir, 'footprints')
      os.makedirs(footprints_dir, exist_ok=True)
      
      # Save in shapefile format
      footprints_shp_path = os.path.join(footprints_dir, filename + '_footprints.shp')
      footprints_gdf.to_file(footprints_shp_path)
      dsm_logger.info(f"Saved building footprints to {footprints_shp_path}")
      
      # Optional: also save in GeoJSON for greater compatibility
      footprints_geojson_path = os.path.join(footprints_dir, filename + '_footprints.geojson')
      footprints_gdf.to_file(footprints_geojson_path, driver='GeoJSON')
      dsm_logger.debug(f"Saved building footprints to {footprints_geojson_path}")
    else:
      dsm_logger.warning(f"No building footprints extracted for {filename}")
      
  except Exception as e:
    dsm_logger.error(f"Error extracting building footprints for {filename}: {str(e)}")
    dsm_logger.debug(f"Error details:", exc_info=True)

  dsm_logger.info(f"Saved {output_tif_path} and {sem_tif_path} with georeference")

# Performances stats
if tilesLen > 0:
    if total_mse > 0:
        dsm_logger.info("Final MSE loss  : " + str(total_mse/tilesLen))
    if total_mae > 0:
        dsm_logger.info("Final MAE loss  : " + str(total_mae/tilesLen))
    if total_rmse > 0:
        dsm_logger.info("Final RMSE loss : " + str(total_rmse/tilesLen))
    
dsm_logger.info(f"DSM generation completed successfully. Results saved to {output_dir}")
