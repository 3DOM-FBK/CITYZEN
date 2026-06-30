#!/usr/bin/env python3
"""
Complete pipeline for orthophoto processing:
1. DSMNet - DSM Generation
2. Roof type classification
3. 3D reconstruction with 3DOM-LOD2-Generator
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
import time
import json
from pathlib import Path

from ndsm_height_calibration import fit_height_calibration

log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=getattr(logging, log_level), 
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Log level set to:{log_level}")

def build_tf_subprocess_env():
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env

class PipelineProcessor:
    def __init__(
        self,
        ortho_path,
        footprints_path,
        output_dir="/workspace/data/output",
        dsm_batch_size=None,
        dsm_step_size_factor=None,
        dsm_refinement_iterations=None,
        dsm_disable_refinement=False,
        dsm_dataset_name="Bologna",
        dsm_checkpoint_dir=None,
        dsm_num_classes=None,
        dsm_building_class_index=1,
        ndsm_height_scale=1.0,
        ndsm_height_offset=0.0,
        ndsm_clamp_min=0.0,
        ndsm_calibration_gt_dir=None,
        ndsm_calibration_mask_dir=None,
        ndsm_calibration_wld_dir=None,
        ndsm_calibration_mode="scale",
        ndsm_calibration_min_buildings=10,
        ndsm_calibration_min_component_pixels=16,
    ):
        self.ortho_path = Path(ortho_path)
        self.footprints_path = Path(footprints_path)
        self.footprints_dir = None
        self.output_dir = Path(output_dir)
        self.dsm_batch_size = dsm_batch_size
        self.dsm_step_size_factor = dsm_step_size_factor
        self.dsm_refinement_iterations = dsm_refinement_iterations
        self.dsm_disable_refinement = dsm_disable_refinement
        self.dsm_dataset_name = dsm_dataset_name
        self.dsm_checkpoint_dir = Path(dsm_checkpoint_dir) if dsm_checkpoint_dir else None
        self.dsm_num_classes = dsm_num_classes
        self.dsm_building_class_index = dsm_building_class_index
        self.ndsm_height_scale = ndsm_height_scale
        self.ndsm_height_offset = ndsm_height_offset
        self.ndsm_clamp_min = ndsm_clamp_min
        self.ndsm_calibration_gt_dir = Path(ndsm_calibration_gt_dir) if ndsm_calibration_gt_dir else None
        self.ndsm_calibration_mask_dir = Path(ndsm_calibration_mask_dir) if ndsm_calibration_mask_dir else None
        self.ndsm_calibration_wld_dir = Path(ndsm_calibration_wld_dir) if ndsm_calibration_wld_dir else None
        self.ndsm_calibration_mode = ndsm_calibration_mode
        self.ndsm_calibration_min_buildings = ndsm_calibration_min_buildings
        self.ndsm_calibration_min_component_pixels = ndsm_calibration_min_component_pixels
        self._calibration_cache = {}
        
        # Create output directories
        self.dsm_output = self.output_dir / "dsm"
        self.rooftype_output = self.output_dir / "rooftype"
        self.models_output = self.output_dir / "3d_models"
        self.calibration_output = self.output_dir / "calibration"
        
        for directory in [self.dsm_output, self.rooftype_output, self.models_output, self.calibration_output]:
            directory.mkdir(parents=True, exist_ok=True)

    def _merged_output_paths(self, ortho_name):
        merged_obj = self.models_output / f"{ortho_name}_buildings.obj"
        merged_ply = self.models_output / f"{ortho_name}_buildings.ply"
        merged_cityjson = self.models_output / f"{ortho_name}_buildings.city.json"
        return merged_obj, merged_ply, merged_cityjson

    def _merged_outputs_exist(self, ortho_name):
        merged_obj, merged_ply, merged_cityjson = self._merged_output_paths(ortho_name)
        return merged_obj.exists() and merged_ply.exists() and merged_cityjson.exists()

    def _cleanup_3dom_intermediates(self, ortho_name, ortho_output_dir=None):
        stale_points_file = self.models_output / f"{ortho_name}_dsm_points.npy"
        if stale_points_file.exists():
            stale_points_file.unlink()
            logger.info("Removed stale intermediate file: %s", stale_points_file.name)

        stale_grid_file = self.models_output / f"{ortho_name}_height_grid.npz"
        if stale_grid_file.exists():
            stale_grid_file.unlink()
            logger.info("Removed temporary height grid: %s", stale_grid_file.name)

        if ortho_output_dir and ortho_output_dir.exists():
            shutil.rmtree(ortho_output_dir)
            logger.info("Removed temporary per-building meshes: %s", ortho_output_dir.name)

    def _height_calibration_enabled(self):
        return self.ndsm_calibration_gt_dir is not None and self.ndsm_calibration_wld_dir is not None

    def _resolve_footprints_dir(self):
        if not self.footprints_path.exists():
            logger.error("Footprints path does not exist: %s", self.footprints_path)
            return None

        if self.footprints_path.is_dir():
            shapefiles = sorted(self.footprints_path.glob("*.shp"))
            if not shapefiles:
                logger.error("No shapefile (.shp) found in footprints directory: %s", self.footprints_path)
                return None
            logger.info("Using supplied footprints directory: %s", self.footprints_path)
            logger.info("Shapefiles found: %s", [path.name for path in shapefiles])
            return self.footprints_path

        if self.footprints_path.suffix.lower() == ".shp":
            logger.info("Using supplied footprint shapefile: %s", self.footprints_path)
            return self.footprints_path.parent

        logger.error("Footprints path must be a directory of .shp files or a single .shp file: %s", self.footprints_path)
        return None

    def _get_height_calibration_for_dsm(self, dsm_path):
        cache_key = str(dsm_path)
        if cache_key in self._calibration_cache:
            return self._calibration_cache[cache_key]

        fallback = {
            "scale": float(self.ndsm_height_scale),
            "offset": float(self.ndsm_height_offset),
            "source": "fallback",
            "report_path": None,
        }

        if not self._height_calibration_enabled():
            logger.info(
                "Height calibration disabled for %s; using default reconstruction scale %.3f and offset %.3f",
                Path(dsm_path).name,
                fallback["scale"],
                fallback["offset"],
            )
            self._calibration_cache[cache_key] = fallback
            return fallback

        report_path = self.calibration_output / f"{Path(dsm_path).stem}_height_calibration.json"
        try:
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = fit_height_calibration(
                    pred_raster_path=dsm_path,
                    reference_ndsm_dir=self.ndsm_calibration_gt_dir,
                    reference_world_dir=self.ndsm_calibration_wld_dir,
                    reference_mask_dir=self.ndsm_calibration_mask_dir,
                    mode=self.ndsm_calibration_mode,
                    min_buildings=self.ndsm_calibration_min_buildings,
                    min_component_pixels=self.ndsm_calibration_min_component_pixels,
                )
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            calibration = {
                "scale": float(report["scale"]),
                "offset": float(report["offset"]),
                "source": report.get("mode", "calibration"),
                "report_path": str(report_path),
            }
            logger.info(
                "Height calibration for %s: scale=%.4f offset=%.4f (%s, %d buildings)",
                Path(dsm_path).name,
                calibration["scale"],
                calibration["offset"],
                calibration["source"],
                int(report.get("building_count", 0)),
            )
        except Exception as exc:
            logger.warning(
                "Height calibration failed for %s: %s. Falling back to scale=%.3f offset=%.3f",
                Path(dsm_path).name,
                exc,
                fallback["scale"],
                fallback["offset"],
            )
            calibration = fallback

        self._calibration_cache[cache_key] = calibration
        return calibration

    def run_dsmnet(self):
        logger.info("Step 1: DSM generation with DSMNet...")
        
        # Handle steps to not redo everything 
        logger.info(f"Search for existing files in: {self.dsm_output}")
        dsm_files = list(self.dsm_output.glob("*_dsm.tif"))
        semantic_files = list(self.dsm_output.glob("*_semantic.tif"))
        
        logger.info(f"DSM files found: {len(dsm_files)}, semantic files found: {len(semantic_files)}")
        if dsm_files:
            logger.info(f"DSM files: {[f.name for f in dsm_files]}")
        if semantic_files:
            logger.info(f"Semantic files:{[f.name for f in semantic_files]}")
        
        # User footprints are mandatory, so the pipeline only needs DSM rasters
        # from DSMNet for height information.
        if dsm_files:
            logger.info(f"DSMNet has already generated the necessary DSMs:")
            logger.info(f"  - DSM file found: {dsm_files[0].name}")
            logger.info("On to the next stage...")
            return True
        
        # Find orthos
        image_files = []
        for ext in ['*.jp2', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(self.ortho_path.glob(ext)))
        
        if not image_files:
            logger.error(f"No images found in the directory {self.ortho_path}")
            logger.error(f"Supported formats: jp2, tif, tiff, jpg, jpeg, png")
            return False
            
        logger.info(f"Images found: {len(image_files)} files in {self.ortho_path}")
        
        try:
            debug_option = []
            if logger.getEffectiveLevel() <= logging.DEBUG:
                debug_option = ["--debug"]
                
            cmd = [
                "python", "/workspace/DSMNet/test_dsm.py",
                "--input_path", str(self.ortho_path),
                "--output_path", str(self.dsm_output)
            ] + debug_option

            if self.dsm_batch_size is not None:
                cmd.extend(["--batch_size", str(self.dsm_batch_size)])
            if self.dsm_step_size_factor is not None:
                cmd.extend(["--step_size_factor", str(self.dsm_step_size_factor)])
            if self.dsm_refinement_iterations is not None:
                cmd.extend(["--refinement_iterations", str(self.dsm_refinement_iterations)])
            if self.dsm_disable_refinement:
                cmd.append("--disable_refinement")
            if self.dsm_dataset_name:
                cmd.extend(["--dataset_name", str(self.dsm_dataset_name)])
            if self.dsm_checkpoint_dir is not None:
                cmd.extend(["--checkpoint_dir", str(self.dsm_checkpoint_dir)])
            if self.dsm_num_classes is not None:
                cmd.extend(["--num_classes", str(self.dsm_num_classes)])
            if self.dsm_building_class_index is not None:
                cmd.extend(["--building_class_index", str(self.dsm_building_class_index)])
            if self.ndsm_clamp_min is not None:
                cmd.extend(["--dsm_clamp_min", str(self.ndsm_clamp_min)])
            
            logger.info(f"DSMNet control: {' '.join(cmd)}")
            
            logger.info("=== Start of DSMNet logs===")
            result = subprocess.run(
                cmd, 
                text=True, 
                cwd="/workspace/DSMNet",
                env=build_tf_subprocess_env()
            )
            logger.info("=== End of DSMNet logs===")
            
            if result.returncode != 0:
                logger.error(f"DSMNet ended with an error code: {result.returncode}")
                return False
                
            logger.info("DSMNet successfully completed")
            return True
            
        except Exception as e:
            logger.error(f"Error when running DSMNet:{e}")
            return False

    def run_rooftype_classification(self):
        logger.info("Step 2: Classification of roof types...")
        
        classification_files = list(self.rooftype_output.glob("*.tif")) + list(self.rooftype_output.glob("*.jpg")) + list(self.rooftype_output.glob("*.png"))
        
        if classification_files:
            logger.info(f"Roof classification already successfully completed:")
            for file in classification_files[:3]:
                logger.info(f"  - {file.name}")
            if len(classification_files) > 3:
                logger.info(f"  - ... et {len(classification_files) - 3} other files")
            logger.info("On to the next stage....")
            return True
        
        try:
            footprints_to_use = self.footprints_dir or self._resolve_footprints_dir()
            if footprints_to_use is None:
                return False
            
            # Execute classification of roofs
            cmd = [
                "python", "/workspace/building_rooftype_classification/orthophoto_inference.py",
                "--input_dir", str(self.ortho_path),
                "--footprints_dir", str(footprints_to_use),
                "--dsm_dir", str(self.dsm_output),
                "--output_dir", str(self.rooftype_output),
                "--model_path", "/workspace/building_rooftype_classification/models/best_fine_tuned_vgg16.keras",
                "--visualize"
            ]
            
            logger.info(f"Order classification: {' '.join(cmd)}")
            
            logger.info("=== Start of roof classification logs ===")
            result = subprocess.run(
                cmd, 
                text=True, 
                cwd="/workspace/building_rooftype_classification",
                env=build_tf_subprocess_env()
            )
            logger.info("=== End of roof classification logs ===")
            
            if result.returncode != 0:
                logger.error(f"The roof classification ended with an error code: {result.returncode}")
                return False
                
            logger.info("Roof classification successfully completed")
            return True
            
        except Exception as e:
            logger.error(f"Error when classifying roofs:{e}")
            return False

    def run_3dom_reconstruction(self):
        logger.info("Step 3: 3D reconstruction with 3DOM-LOD2-Generator...")
        logger.info("3DOM height source: raster height grid")
        
        # Check for individual orthophoto reconstruction directories instead of global files
        # This allows for proper multi-orthophoto processing
        
        try:
            classified_vectors = sorted(self.rooftype_output.glob("*_classified.geojson"))
            vector_format = "GeoJSON"

            if not classified_vectors:
                classified_vectors = sorted(self.rooftype_output.glob("*_classified.shp"))
                vector_format = "Shapefile"

            if not classified_vectors:
                logger.error("No classified GeoJSON or shapefile found in the rooftype directory")
                return False
            
            logger.info(f"Using {vector_format} inputs for 3DOM reconstruction")
            logger.info(f"Treatment of {len(classified_vectors)} classified vector files:")
            for vector_file in classified_vectors:
                logger.info(f"  - {vector_file.name}")
            
            # Only use actual DSM rasters for 3DOM reconstruction.
            dsm_files = sorted(self.dsm_output.glob("*_dsm.tif"))
            logger.info(f"DSM files available:{len(dsm_files)}")
            for dsm in dsm_files:
                logger.info(f"  - {dsm.name}")

            if not dsm_files:
                logger.error("No *_dsm.tif files found for 3DOM reconstruction")
                return False
            
            success_count = 0
            
            # Process each classified vector file
            for input_vector in classified_vectors:
                logger.info(f"=== Treatment of {input_vector.name} ===")
                
                ortho_name = input_vector.stem.replace('_classified', '')
                logger.info(f"Name of the orthophoto: {ortho_name}")
                
                merged_obj, merged_ply, merged_cityjson = self._merged_output_paths(ortho_name)
                ortho_output_dir = self.models_output / ortho_name

                # Prefer the final merged outputs as the signal that this orthophoto is complete.
                if self._merged_outputs_exist(ortho_name):
                    logger.info(
                        "Orthophoto %s already has merged outputs (%s, %s, %s), skipping reconstruction",
                        ortho_name,
                        merged_obj.name,
                        merged_ply.name,
                        merged_cityjson.name,
                    )
                    self._cleanup_3dom_intermediates(ortho_name, ortho_output_dir)
                    success_count += 1
                    continue

                if ortho_output_dir.exists():
                    existing_ply_files = list(ortho_output_dir.glob("*.ply"))
                    if existing_ply_files:
                        logger.info(
                            "Orthophoto %s already has %d temporary per-building meshes, "
                            "skipping Blender regeneration and reusing them for merge",
                            ortho_name,
                            len(existing_ply_files),
                        )
                        success_count += 1
                        continue
                
                # Find corresponding DSM file
                corresponding_dsm = None
                height_grid_file = None
                
                # Match the reconstruction input to the DSM generated for the same orthophoto.
                dsm_by_base = {dsm_file.stem.replace('_dsm', ''): dsm_file for dsm_file in dsm_files}
                corresponding_dsm = dsm_by_base.get(ortho_name)

                if not corresponding_dsm:
                    compatible_dsms = [
                        dsm_file for dsm_base, dsm_file in dsm_by_base.items()
                        if ortho_name in dsm_base or dsm_base in ortho_name
                    ]

                    if len(compatible_dsms) == 1:
                        corresponding_dsm = compatible_dsms[0]
                        logger.warning(
                            f"No exact DSM match found for {ortho_name}, using compatible file {corresponding_dsm.name}"
                        )
                    elif len(compatible_dsms) > 1:
                        logger.error(
                            f"Multiple DSM candidates found for {ortho_name}: {[dsm.name for dsm in compatible_dsms]}"
                        )
                        continue
                    else:
                        logger.error(f"No matching *_dsm.tif found for {ortho_name}")
                        continue
                
                if corresponding_dsm:
                    logger.info(f"Using the DSM file: {corresponding_dsm.name}")
                    height_calibration = self._get_height_calibration_for_dsm(corresponding_dsm)
                    current_height_scale = height_calibration["scale"]
                    current_height_offset = height_calibration["offset"]
                    if height_calibration.get("report_path"):
                        logger.info(f"Calibration report: {height_calibration['report_path']}")
                    logger.info("Using DSM raster directly for per-building height queries")

                    height_grid_file = self.models_output / f"{ortho_name}_height_grid.npz"
                    if not height_grid_file.exists():
                        convert_cmd = [
                            "python", "/workspace/3dom-lod2-generator/tool/dsm_to_height_grid.py",
                            "-i", str(corresponding_dsm),
                            "-o", str(height_grid_file),
                            "--z_scale", str(current_height_scale),
                            "--z_offset", str(current_height_offset),
                            "--clamp_min", str(self.ndsm_clamp_min),
                        ]
                        logger.info(f"Height-grid conversion command: {' '.join(convert_cmd)}")
                        convert_result = subprocess.run(
                            convert_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            universal_newlines=True,
                        )
                        if convert_result.stdout:
                            for line in convert_result.stdout.strip().split('\n'):
                                if line.strip():
                                    logger.info(f"Height Grid: {line}")
                        if convert_result.returncode != 0:
                            logger.error(
                                "Height-grid conversion failed for %s (code %s)",
                                ortho_name,
                                convert_result.returncode,
                            )
                            continue
                    else:
                        logger.info(f"Existing height grid file: {height_grid_file.name}")

                # 3D Reconstruction for this shapefile
                cmd = [
                    "blender", "--background", "--python", "/workspace/3dom-lod2-generator/tool/blender_main.py",
                    "--",
                    "-i", str(input_vector),
                    "-o", str(self.models_output),
                    "--ortho_name", ortho_name,
                ]

                if height_grid_file and height_grid_file.exists():
                    cmd.extend(["--dsm_grid", str(height_grid_file)])
                elif corresponding_dsm:
                    cmd.extend([
                        "--dsm_raster", str(corresponding_dsm),
                        "--ndsm_height_scale", str(current_height_scale),
                        "--ndsm_height_offset", str(current_height_offset),
                    ])
                
                logger.info(f"3DOM command: {' '.join(cmd)}")
                logger.info(f"=== Start of 3D reconstruction logs for {ortho_name} ===")
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    cwd="/workspace/3dom-lod2-generator",
                    universal_newlines=True
                )
                
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            if any(keyword in line.lower() for keyword in ['error', 'erreur', 'failed', 'échec']):
                                logger.warning(f"3DOM: {line}")
                            else:
                                logger.debug(f"3DOM: {line}")
                
                logger.info(f"=== End of 3D reconstruction logs for {ortho_name} ===")
                
                if result.returncode != 0:
                    logger.error(f"3DOM reconstruction error for {ortho_name} (code {result.returncode})")
                    continue 
                else:
                    written_meshes = []
                    if ortho_output_dir.exists():
                        written_meshes = list(ortho_output_dir.glob("*.ply"))

                    if not written_meshes:
                        logger.error(
                            "3D reconstruction for %s finished without writing any per-building PLY meshes",
                            ortho_name,
                        )
                        continue

                    logger.info(
                        "3D reconstruction successfully completed for %s with %d temporary meshes",
                        ortho_name,
                        len(written_meshes),
                    )
                    success_count += 1
            
            logger.info(f"3D reconstruction complete: {success_count}/{len(classified_vectors)} réussies")
            return success_count > 0 
                
        except Exception as e:
            logger.error(f"Erreur lors de la reconstruction 3D: {e}")
            return False

    def merge_3d_models(self):
        logger.info("Step 4: Merging 3D models with orthophotos...")
        
        # Find all orthophoto directories
        ortho_dirs = [d for d in self.models_output.iterdir() if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('temp')]
        
        if not ortho_dirs:
            merged_outputs = list(self.models_output.glob("*_buildings.ply"))
            if merged_outputs:
                logger.info("No temporary per-building directories found; merged outputs are already present")
                return True
            logger.warning("No orthophoto directory found for the merger")
            return False
        
        logger.info(f"Orthophoto directories found: {len(ortho_dirs)}")
        for ortho_dir in ortho_dirs:
            logger.info(f"  - {ortho_dir.name}")
        
        success_count = 0
        
        for ortho_dir in ortho_dirs:
            ortho_name = ortho_dir.name
            logger.info(f"=== Merging models for {ortho_name} ===")
            
            # Find PLY files in this orthophoto directory
            ply_files = list(ortho_dir.glob("*.ply"))
            
            if not ply_files:
                logger.warning(f"No PLY files found in{ortho_name}")
                continue
            
            logger.info(f"  {len(ply_files)} PLY files found in {ortho_name}")
            
            # Output OBJ file for this orthophoto
            merged_output = self.models_output / f"{ortho_name}_buildings.obj"
            source_vector = self.rooftype_output / f"{ortho_name}_classified.geojson"
            if not source_vector.exists():
                source_vector = self.rooftype_output / f"{ortho_name}_classified.shp"
            source_raster = self.dsm_output / f"{ortho_name}_dsm.tif"
            
            try:
                cmd = [
                    "python", "/workspace/3dom-lod2-generator/tool/merge.py",
                    "--models_dir", str(ortho_dir),
                    "--output", str(merged_output)
                ]
                if source_vector.exists():
                    cmd.extend(["--source_vector", str(source_vector)])
                if source_raster.exists():
                    cmd.extend(["--source_raster", str(source_raster)])
                
                logger.info(f"Commande fusion pour {ortho_name}: {' '.join(cmd)}")
                logger.info(f"=== Start of 3D fusion logs for {ortho_name} ===")
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    universal_newlines=True
                )
                
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"Fusion {ortho_name}: {line}")
                
                logger.info(f"=== End of 3D fusion logs for{ortho_name} ===")
                
                if result.returncode != 0:
                    logger.error(f"3D fusion error for {ortho_name} (code {result.returncode})")
                    continue
                else:
                    logger.info(f"3D model merge successfully completed for{ortho_name}")
                    logger.info(f"Merged model available: {merged_output.name}")
                    logger.info(f"CityJSON model available: {merged_output.with_suffix('.city.json').name}")
                    self._cleanup_3dom_intermediates(ortho_name, ortho_dir)
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error when merging 3D models for {ortho_name}: {e}")
                continue
        
        logger.info(f"Merger complete: {success_count}/{len(ortho_dirs)} orthophotos successfully merged")
        return success_count > 0

    def run_pipeline(self):
        """Runs the entire pipeline"""
        logger.info("Start-up of the complete pipeline...")
        
        if not self.ortho_path.exists():
            logger.error(f"The orthophoto directory does not exist: {self.ortho_path}")
            return False

        self.footprints_dir = self._resolve_footprints_dir()
        if self.footprints_dir is None:
            logger.error("Footprints are mandatory. Provide --footprints_path with a valid shapefile or shapefile directory.")
            return False
        
        # Step 1: DSMNet
        if not self.run_dsmnet():
            logger.error("Failure of the DSMNet stage")
            return False
        
        # Step 2: Roof classification
        if not self.run_rooftype_classification():
            logger.error("Failure to classify roofs")
            return False
        
        # Step 3: 3D Reconstruction
        if not self.run_3dom_reconstruction():
            logger.error("Failure of 3D reconstruction")
            return False
        
        # Step 4: 3D Fusion
        if not self.merge_3d_models():
            logger.warning("Failure to merge 3D models (non-critical)")
        
        logger.info("Pipeline successfully completed!")
        logger.info(f"Results available in: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Complete orthophoto processing pipeline")
    parser.add_argument("--ortho_path", default="/workspace/data/input/ortho", 
                       help="Path to the orthophotos directory (default:/workspace/data/input/ortho)")
    parser.add_argument("--footprints_path", required=True,
                       help="Path to a footprints directory or a single .shp file (required)")
    parser.add_argument("--output_dir", default="/workspace/data/output", 
                       help="Output directory (default: /workspace/data/output)")
    parser.add_argument("--dsm_batch_size", type=int,
                       help="Batch size for DSMNet crop inference (optional)")
    parser.add_argument("--dsm_step_size_factor", type=float,
                       help="Sliding-window step factor for DSMNet (optional)")
    parser.add_argument("--dsm_refinement_iterations", type=int,
                       help="Number of DSMNet refinement iterations (optional)")
    parser.add_argument("--dsm_disable_refinement", action="store_true",
                       help="Disable DSMNet refinement for faster inference")
    parser.add_argument("--dsm_dataset_name",
                       default=os.environ.get("CITYZEN_DSMNET_DATASET", "Bologna"),
                       help="DSMNet inference profile: Vaihingen, DFC2018, or Bologna")
    parser.add_argument("--dsm_checkpoint_dir",
                       default=os.environ.get("CITYZEN_DSMNET_CHECKPOINT_DIR", "/workspace/DSMNet/checkpoints/Bologna"),
                       help="Optional DSMNet checkpoint directory overriding the default dataset path")
    parser.add_argument("--dsm_num_classes", type=int,
                       default=int(os.environ["CITYZEN_DSMNET_NUM_CLASSES"]) if os.environ.get("CITYZEN_DSMNET_NUM_CLASSES") else 2,
                       help="Optional override for the DSMNet semantic output size")
    parser.add_argument("--dsm_building_class_index", type=int,
                       default=int(os.environ.get("CITYZEN_DSMNET_BUILDING_CLASS_INDEX", "1")),
                       help="Semantic class index used to extract building footprints from DSMNet output")
    parser.add_argument("--ndsm_height_scale", type=float,
                       default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_SCALE", "1.0")),
                       help="Default Step 1 nDSM scale used for 3DOM reconstruction when no fitted calibration is enabled (Bologna default: 1.0)")
    parser.add_argument("--ndsm_height_offset", type=float,
                       default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_OFFSET", "0.0")),
                       help="Optional vertical offset added after nDSM scaling")
    parser.add_argument("--ndsm_clamp_min", type=float,
                       default=float(os.environ.get("CITYZEN_NDSM_CLAMP_MIN", "0.0")),
                       help="Clamp DSM/nDSM values below this threshold before saving rasters and during 3DOM reconstruction")
    parser.add_argument("--ndsm_calibration_gt_dir",
                       default=os.environ.get("CITYZEN_NDSM_CALIBRATION_GT_DIR"),
                       help="Optional reference nDSM patch directory for fitted height calibration; leave unset to keep the default 1.0 runtime scale")
    parser.add_argument("--ndsm_calibration_mask_dir",
                       default=os.environ.get("CITYZEN_NDSM_CALIBRATION_MASK_DIR"),
                       help="Optional reference building-mask patch directory for fitted height calibration")
    parser.add_argument("--ndsm_calibration_wld_dir",
                       default=os.environ.get("CITYZEN_NDSM_CALIBRATION_WLD_DIR"),
                       help="Optional world-file directory describing the reference patch placement")
    parser.add_argument("--ndsm_calibration_mode",
                       choices=["scale", "affine"],
                       default=os.environ.get("CITYZEN_NDSM_CALIBRATION_MODE", "scale"),
                       help="Calibration model used when reference patches are supplied")
    parser.add_argument("--ndsm_calibration_min_buildings", type=int,
                       default=int(os.environ.get("CITYZEN_NDSM_CALIBRATION_MIN_BUILDINGS", "10")),
                       help="Minimum building samples required for automatic height calibration")
    parser.add_argument("--ndsm_calibration_min_component_pixels", type=int,
                       default=int(os.environ.get("CITYZEN_NDSM_CALIBRATION_MIN_COMPONENT_PIXELS", "16")),
                       help="Ignore tiny building-mask components below this size during calibration")
    args = parser.parse_args()
    
    #Print files architecture (very useful in this case I promise)
    logger.info(f"Use of paths:")
    logger.info(f"  - Orthophotos: {args.ortho_path}")
    logger.info(f"  - Footprints: {args.footprints_path}")
    logger.info(f"  - Sortie: {args.output_dir}")
    
    processor = PipelineProcessor(
        ortho_path=args.ortho_path,
        footprints_path=args.footprints_path,
        output_dir=args.output_dir,
        dsm_batch_size=args.dsm_batch_size,
        dsm_step_size_factor=args.dsm_step_size_factor,
        dsm_refinement_iterations=args.dsm_refinement_iterations,
        dsm_disable_refinement=args.dsm_disable_refinement,
        dsm_dataset_name=args.dsm_dataset_name,
        dsm_checkpoint_dir=args.dsm_checkpoint_dir,
        dsm_num_classes=args.dsm_num_classes,
        dsm_building_class_index=args.dsm_building_class_index,
        ndsm_height_scale=args.ndsm_height_scale,
        ndsm_height_offset=args.ndsm_height_offset,
        ndsm_clamp_min=args.ndsm_clamp_min,
        ndsm_calibration_gt_dir=args.ndsm_calibration_gt_dir or None,
        ndsm_calibration_mask_dir=args.ndsm_calibration_mask_dir or None,
        ndsm_calibration_wld_dir=args.ndsm_calibration_wld_dir or None,
        ndsm_calibration_mode=args.ndsm_calibration_mode,
        ndsm_calibration_min_buildings=args.ndsm_calibration_min_buildings,
        ndsm_calibration_min_component_pixels=args.ndsm_calibration_min_component_pixels,
    )
    
    success = processor.run_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
