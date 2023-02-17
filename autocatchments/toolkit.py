from typing import Tuple

import matplotlib
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator, SinkFillerBarnes
from landlab.components.flow_accum.flow_accum_bw import (
    find_drainage_area_and_discharge,
    make_ordered_node_array,
)
from matplotlib.colors import LogNorm
from osgeo import gdal

"""Generic functions useful for analysing drainage networks."""


def viz_drainage_area(grid: RasterModelGrid):
    """ "Visualises drainage area logarithmically.
    Args:
        model_grid (RasterModelGrid): LandLab grid with drainage routed across it.

    Returns:
        None

    Produces instance of matplotlib.plt"""

    matplotlib.pyplot.imshow(
        grid.at_node["drainage_area"].reshape(grid.shape),
        norm=LogNorm(),
        extent=[
            grid.xy_of_lower_left[0],
            grid.xy_of_lower_left[0] + grid.shape[1] * grid.dx,
            grid.xy_of_lower_left[1],
            grid.xy_of_lower_left[1] + grid.shape[0] * grid.dy,
        ],
        cmap="Greys",
    )
    cb = matplotlib.pyplot.colorbar()
    cb.set_label("Drainage Area/units^2")
    matplotlib.pyplot.xlabel("x / units")
    matplotlib.pyplot.ylabel("y / units")


def get_gdal_grid_metadata(ds: gdal.Dataset) -> Tuple[tuple, tuple]:

    geotransform = ds.GetGeoTransform()
    top_left_x = geotransform[0]
    top_left_y = geotransform[3]
    dx = geotransform[1]
    dy = geotransform[5]
    ny = ds.RasterYSize
    lower_left_xy = (top_left_x, top_left_y + ny * dy)
    xy_spacing = (dx, abs(dy))
    return (xy_spacing, lower_left_xy)


def load_topo(path):
    """Turns a topographic data file (as .asc) into a LandLab model grid with drainage routed across it.

    Args:
        path (string): Path to the geospatial raster file.
    Returns:
        RasterModelGrid: An initialised LandLab grid with sinks filled and drainage routed across it.

    TODO: Handle no-data (e.g., coastal values) properly
    TODO: Initiate the model nodes with a cell area
    """

    elev_arr, ds = read_geo_file(path)
    dx_dy, ll_xy = get_gdal_grid_metadata(ds)
    model_grid = RasterModelGrid(elev_arr.shape, xy_spacing=dx_dy, xy_of_lower_left=ll_xy)
    model_grid.add_field("topographic__elevation", elev_arr.flatten().astype(float))
    print("Filling sinks (can be slow)")
    sb = SinkFillerBarnes(model_grid, ignore_overfill=True)
    sb.run_one_step()
    print("Running flow-routing")
    frr = FlowAccumulator(model_grid, flow_director="FlowDirectorD8")
    frr.run_one_step()
    return model_grid


def read_geo_file(filename: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """Reads a geospatial file"""
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds


def get_neighbour_indices(ind: int, array: np.ndarray) -> list[list]:
    """Gets indices of the nodes surrounding node `ind' (and itself) in array `array'.
    Neighbours given in clockwise order:  [self, r,br,b,bl,l,tl,r,tr].
    Assumes index lies as a core node."""
    # Assumes index lies within grid
    nx = array.shape[1]
    # [self,r,br,b,bl,l,tl,t,tr]
    neighbours = [
        ind,
        ind + 1,
        ind + nx + 1,
        ind + nx,
        ind + nx - 1,
        ind - 1,
        ind - nx - 1,
        ind - nx,
        ind - nx + 1,
    ]
    return neighbours


def convert_esri_d8(d8_arc: np.ndarray) -> np.ndarray:
    """Convert d8 from arc directions [2^0 -> 2^7, & 0 for sinks]
    to directions encoded by:
    [sink,r,br,b,bl,l,tl,t,tr]  = [0,1,2,3,4,5,6,7,8]"""

    d8_arc[d8_arc == 0] = 0.5
    d8 = np.log2(d8_arc) + 1
    return d8.astype(int)


def d8_to_receivers(d8: np.ndarray) -> np.ndarray:
    """Converts 2D array of d8 flow-directions encoded as:
    [sink,r,br,b,bl,l,tl,t,tr]  = [0,1,2,3,4,5,6,7,8], and returns
    the receiver node array. See `convert_esri_d8'.

    TODO: Speed up by replacing transposition of core_neighbours"""
    # Initialise receivers as self (i.e. sinks)
    receivers = np.arange(d8.size).reshape(d8.shape)
    # Get list of neighbours for each core node (i.e. not boundaries)
    core_neighbours = np.transpose(
        [get_neighbour_indices(i, receivers) for i in receivers[1:-1, 1:-1].flatten()]
    )
    # Use list of neighbours + D8 to select receiver node
    receivers[1:-1, 1:-1] = np.choose(d8[1:-1, 1:-1].flatten(), core_neighbours).reshape(
        receivers[1:-1, 1:-1].shape
    )
    return receivers.flatten()


def load_d8(path: str) -> RasterModelGrid:
    """Reads a (generic) geospatial file containing ESRI D8
    flow directions (e.g., 0, 1, ..., 128) and calculates
    drainage network, assigning this to a landlab RasterModelGrid.
    """

    print("Reading infile")
    d8_arc, ds = read_geo_file(path)
    dx_dy, ll_xy = get_gdal_grid_metadata(ds)
    print("Converting D8 encoding")
    d8 = convert_esri_d8(d8_arc)
    print("Calculating receivers")
    receivers = d8_to_receivers(d8)
    print("Generating drainage stack")
    ordered_nodes = make_ordered_node_array(receivers)
    print("Initialising model grid")
    grid = RasterModelGrid(d8_arc.shape, xy_spacing=dx_dy, xy_of_lower_left=ll_xy)
    _ = grid.add_field("flow__upstream_node_order", ordered_nodes)
    _ = grid.add_field("flow__receiver_node", receivers)
    print("Calculating drainage area")
    a, _ = find_drainage_area_and_discharge(
        ordered_nodes, receivers, node_cell_area=grid.dx * grid.dy
    )
    _ = grid.add_field("drainage_area", a)
    return grid


def load_from_node_arrays(
    path_to_receiver_nodes: str,
    path_to_ordered_nodes: str,
    shape: Tuple[int, int],
    xy_spacing: Tuple[float, float] = (1.0, 1.0),
    xy_of_lower_left: Tuple[float, float] = (0, 0),
) -> RasterModelGrid:
    """Inititates a model grid from precalculated node arrays to speed up
    computation"""

    receivers = np.loadtxt(path_to_receiver_nodes).astype(np.int64)
    ordered_nodes = np.loadtxt(path_to_ordered_nodes).astype(np.int64)
    print("Initialising model grid")
    grid = RasterModelGrid(shape, xy_spacing=xy_spacing, xy_of_lower_left=xy_of_lower_left)
    _ = grid.add_field("flow__upstream_node_order", ordered_nodes)
    _ = grid.add_field("flow__receiver_node", receivers)
    print("Calculating drainage area")
    a, _ = find_drainage_area_and_discharge(
        ordered_nodes, receivers, node_cell_area=grid.dx * grid.dy
    )
    _ = grid.add_field("drainage_area", a)
    return grid


def geographic_coords_to_model_xy(
    xy_coords: Tuple[np.ndarray, np.ndarray], grid: RasterModelGrid
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts geographical coordinates (from lower left) into model
    grid coordinates (from upper left)"""
    xy_of_upper_left = grid.xy_of_lower_left[0], grid.xy_of_lower_left[1] + grid.dy * grid.shape[0]
    x = (xy_coords[0] - xy_of_upper_left[0]) / grid.dx
    y = (xy_of_upper_left[1] - xy_coords[1]) / grid.dy
    return x, y


def model_xy_to_geographic_coords(
    model_xy_coords: Tuple[np.ndarray, np.ndarray], grid: RasterModelGrid
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts model grid coordinates (from upper left) to geographical coordinates
    (from lower left)"""
    xy_of_upper_left = grid.xy_of_lower_left[0], grid.xy_of_lower_left[1] + grid.dy * grid.shape[0]
    x = xy_of_upper_left[0] + model_xy_coords[0] * grid.dx
    y = xy_of_upper_left[1] - model_xy_coords[1] * grid.dy
    return x, y
