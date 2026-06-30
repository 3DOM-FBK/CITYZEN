import logging
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from shapefile.converter import create_mesh_from_polygon
from io_utils.exporter import apply_global_shift, export_mesh_ply
from modeling.roofs.flat import create_flat_roof
from modeling.roofs.gabled import create_gabled_roof
import modeling.blender_ops as blender_ops
