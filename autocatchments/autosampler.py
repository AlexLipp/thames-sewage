import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autocatchments.toolkit import (
    model_xy_to_geographic_coords,
    viz_drainage_area,
    geographic_coords_to_model_xy,
)

"""Functions useful for processing samples wihch are gathered on a drainage network 
(e.g., geochemical observations etc.)"""


def fast_delete(x, elements):
    """Removes `elements` from numpy array by value faster than list.remove
    Args:
        x (array): The array to remove elements from.
        elements (array like): The elements to remove from x
    Returns:
        (array): The array with elements removed
    """
    indices = np.ravel([np.where(x == e) for e in elements])
    return np.delete(x, indices)


def get_sample_nodes_by_area(model_grid, target_area):
    """Finds sample sites which best divide DEM into ~equal sub-catchments
    Args:
        model_grid (RasterModelGrid): The landscape to partition. This is
        a LandLab RasterModelGrid object that must have flow routed across it using D8 method.
        It is recommended that sinks are filled too to allow for continuous flow paths.

        target_area: The area (in units of the model grid NOT number of nodes) which basins should be
        larger than. Note that this is a *minimum* value and basins may be this size (or greater)
        but no smaller.

    Returns:
        {int: list of ints}: Dictionary, the keys of which are the node IDs of the identified sample sites.
        The items for each key are the IDs of the nodes in the subcatchment delineated by that sample.
        Node IDs can be turned into coordinates using `process_output_dict` or `np.unravel_index`.
    """

    print("~~~~~~~ Beginning Calculation ~~~~~~~")
    # Node array contining downstream-to-upstream ordered list of node
    ordered_nodes = model_grid.at_node["flow__upstream_node_order"]
    receiver_at_node = model_grid.at_node["flow__receiver_node"]
    cell_area = model_grid.dx * model_grid.dy
    nodes_per_samp = target_area / cell_area

    uV = ordered_nodes.copy()  # unvisited nodes
    print("Removing catchments smaller than target area")
    # Remove nodes from uV if they are within catchments smaller than target (as they will never be visited)
    num_sinks = len(np.where(model_grid.at_node["flow__sink_flag"])[0])
    counter = 0
    for sink in np.where(model_grid.at_node["flow__sink_flag"])[0]:
        if counter % 1000 == 0:
            print(
                "\t Processed sink",
                counter,
                "of",
                num_sinks,
                ",",
                datetime.now().strftime("%H:%M:%S"),
            )
        if model_grid.at_node["drainage_area"][sink] < target_area:
            i = np.where(ordered_nodes == sink)[0][0]
            bad_nodes = [sink]
            for up_node in ordered_nodes[i + 1 :]:
                if receiver_at_node[up_node] in bad_nodes:
                    bad_nodes.append(up_node)
                else:
                    uV = fast_delete(uV, bad_nodes)
                    break
        counter += 1
    sample_nodes = {}
    counter = 0
    print("Target area = ", nodes_per_samp)
    print("Looping through all unvisited nodes upstream to downstream")
    initial_len = len(uV)
    # Iterate through nodes from upstream to downstream
    for i in np.arange(initial_len - 1, -1, -1):
        if counter % 1000 == 0:
            print(
                "\t Processing node",
                counter,
                "of",
                initial_len,
                ",",
                datetime.now().strftime("%H:%M:%S"),
            )
        node = uV[i]  # Node in network
        # Initiate list of unvisited nodes that are in the upstream catchment of node in question
        unvis_up_nodes = [node]
        # Loop through unvisited nodes upstream
        for new_up_node in uV[i + 1 :]:
            # If this node drains to a node in our list we add it to the list
            # Hence we progressively go upstream building the subcatchment
            if receiver_at_node[new_up_node] in unvis_up_nodes:
                unvis_up_nodes.append(new_up_node)
            # When we find a node not in the subcatchment we stop as we have reached a drainage divide
            else:
                break
        # If number of nodes in new subcatchment greater than threshold we add it to output
        if len(unvis_up_nodes) > nodes_per_samp:
            print("\t * Found a sample locality *")
            sample_nodes[node] = unvis_up_nodes  # Add node to list with corresponding catchment
            uV = fast_delete(
                uV, unvis_up_nodes
            )  # Remove the new catchment  from array of unvisited nodes
        counter += 1
    print("Found", len(sample_nodes.keys()), "sample localities")
    area_sizes = [len(areas) * cell_area for _, areas in sample_nodes.items()]
    mean, std = np.mean(area_sizes), np.std(area_sizes)
    print("Average area per basin = ", mean, "+/-", std)
    print("~~~~~~~ Finished Calculation ~~~~~~~")
    return sample_nodes


def process_output_dict(node_catchment_dict, model_grid):
    """Reformats the output dictionary.
    Args:
        node_catchment_dict ({int: list of ints}): Output from `get_sample_nodes_by_area`
        model_grid (RasterModelGrid): LandLab grid with drainage routed across it.
    Returns:
        np.array(N,3): 3 column table output with cols: Sample ID, x-coordinate, y-coordinate. Has
        number of rows equal to number of allocated samples.

        np.array(nx,ny): Map of the identified sub-catchments, with ID corresponding to the Sample ID in
        associated table. 2D array with same dimensions as model grid/DEM. Areas not covered by sample sites
        given the NaN value of -999.

    """

    out_area = np.zeros(model_grid.shape).flatten() - 999
    N = 1
    Ns, model_xs, model_ys = [], [], []
    for node, upst_nodes in node_catchment_dict.items():
        y, x = np.unravel_index(node, model_grid.shape)
        Ns += [N]
        model_xs += [x]
        model_ys += [y]
        out_area[upst_nodes] = N
        N += 1
    out_area = out_area.reshape(model_grid.shape)

    xs, ys = model_xy_to_geographic_coords((np.array(model_xs), np.array(model_ys)), model_grid)
    return (np.array([Ns, xs, ys]).T, out_area)


def save_autosampler_results(locs, areas, model_grid, out_dir):
    """Saves output as files with appropriate names.
    Args:
    locs (np.array(N,3)): 3 column table output with cols: Sample ID, x-coordinate, y-coordinate. Has
            number of rows equal to number of allocated samples. See `process_output_dict`

    areas (np.array(nx,ny)): Map of the identified sub-catchments, with ID corresponding to the Sample ID in
        associated table. 2D array with same dimensions as model grid/DEM. Areas not covered by sample sites
        given the NaN value of -999. See `process_output_dict`

    model_grid (RasterModelGrid): LandLab grid with drainage routed across it.

    Returns:
        None

    Produces two files:
        1. "sample_sites.csv" a file containing the sample site localities, given in `locs` input
        2. "optimal_area_IDs.asc" a map of the delineated sub-catchments, given in `areas` input. This is an
        ESRI ASCII raster file format appropriate for use in most GIS software.

    """
    np.savetxt(
        out_dir + "/optimal_sample_sites.csv",
        X=locs,
        delimiter=",",
        header="Area ID, x, y",
        comments="",
    )
    if os.path.exists(out_dir + "/optimal_area_IDs.asc"):  # Allows over-writing of .asc files
        os.remove(out_dir + "/optimal_area_IDs.asc")
    _ = model_grid.add_field("optimal_area_IDs", np.flipud(areas))
    model_grid.save(out_dir + "/optimal_area_IDs.asc", names="optimal_area_IDs")


def viz_sample_site_results(locs, areas, grid):
    """Visuaslises identified sample localities and associated sub-catchments.
    Args:
        locs (np.array(N,3)): 3 column table output with cols: Sample ID, x-coordinate, y-coordinate. Has
                number of rows equal to number of allocated samples. See `process_output_dict`

        areas (np.array(nx,ny)): Map of the identified sub-catchments, with ID corresponding to the Sample ID in
            associated table. 2D array with same dimensions as model grid/DEM. Areas not covered by sample sites
            given the NaN value of -999. See `process_output_dict`

        model_grid (RasterModelGrid): LandLab grid with drainage routed across it.

    Returns:
        None

    Produces instance of matplotlib.plt
    """
    plt.figure(figsize=(12, 5))
    areas[areas < 0] = np.nan
    plt.subplot(1, 2, 1)
    plt.imshow(
        areas,
        cmap="nipy_spectral",
        extent=[
            grid.xy_of_lower_left[0],
            grid.xy_of_lower_left[0] + grid.shape[1] * grid.dx,
            grid.xy_of_lower_left[1],
            grid.xy_of_lower_left[1] + grid.shape[0] * grid.dy,
        ],
    )
    cb = plt.colorbar()
    cb.set_label("Area ID")
    plt.title("Sample areas")
    plt.scatter(x=locs[:, 1], y=locs[:, 2], c="black", marker="x", s=50)
    plt.xlabel("x / units")
    plt.ylabel("y / units")
    plt.subplot(1, 2, 2)
    viz_drainage_area(grid)
    plt.scatter(x=locs[:, 1], y=locs[:, 2], c="red", marker="x", s=50)
    plt.tight_layout()


def snap_to_drainage(model_grid, sample_sites, threshold):
    """Moves sample sites to nearest channel node with area greater than threshold
    Args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.load_topo and load_d8)

        sample_sites (Nx2 float array): Coordinates [x,y] (in units of the base DEM, e.g., km) of sampled
        localities. These will not necessarily lie on a modelled channel in the DEM.

        threshold (float): Drainage area (in units of base DEM) threshold, above which we define
        a node to be a `channel'. Nodes with area less than this cannot be snapped to.

    Returns:

        snapped (Nx2 float array): Coordinates [x,y] of the snapped, drainage aligned sample sites
        (in units of the base DEM, e.g., km).
    """

    channels = model_grid.at_node["drainage_area"] > threshold
    channels_y_ind, channels_x_ind = np.unravel_index(
        np.ravel(np.where(channels)), model_grid.shape
    )
    channels_model_xy = np.array([channels_x_ind, channels_y_ind]).T

    samps_geog_x = sample_sites[:, 0]
    samps_geog_y = sample_sites[:, 1]

    samps_model_x, samps_model_y = geographic_coords_to_model_xy(
        (samps_geog_x, samps_geog_y), model_grid
    )
    samps_model_xy = np.array([samps_model_x,samps_model_y]).T
    snapped = np.zeros(sample_sites.shape)
    for i in range(sample_sites.shape[0]):
        sample = samps_model_xy[i,:]
        diff = channels_model_xy - sample
        closest_channel_coords = channels_model_xy[np.argmin(np.sum(diff**2, axis=1)),:]
        snapped[i, :] = closest_channel_coords
    return snapped


def coords_to_node_ids(model_grid, coordinates):
    """Converts model grid coordinates into node IDs
    args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.process_topo

        coordinates (Nx2 float array): Coordinates on model grid for which nodes are being found

    returns:
        nodes: corresponding node IDs for each coordinate pair"""

    n_samples = coordinates.shape[0]
    nodes = np.zeros(n_samples,dtype=int)
    for i in np.arange(n_samples):
        x, y = coordinates[i, 0], coordinates[i, 1]
        x_ind = int(np.rint(x))
        y_ind = int(np.rint(y))
        nodes[i] = np.ravel_multi_index((y_ind, x_ind), model_grid.shape)
    return nodes


def get_subcatchments(model_grid, target_nodes, sample_names):
    """Subdivides a drainage network into subcatchments based on provided sample nodes

    args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.process_topo

        target_nodes (int array): IDs of the sample nodes

        sample_names (str array): The sample name for each sample site.
        This *MUST* have the same order as target_nodes

    Returns:
        sample_catchments (nested dict{{}}): Returns a nested dictionary structure
        describing the information about the sub-catchment. The key 'areaID' is the
        integer to which each sub-catchment is assigned going from 1 (upstream) to N
        (most downstream). sample_catchments[areaID] returns a dictionary which stores:
        1) the SampleName for that sample site (using key "SampleName"); 2) the nodeIDs of
        the catchment nodes (using key "catchment_nodes") and 3) the nodeID of the sample
        site (using key "node")"""

    ordered_nodes = model_grid.at_node["flow__upstream_node_order"]
    receiver_at_node = model_grid.at_node["flow__receiver_node"]
    uV = ordered_nodes.copy()  # unvisited nodes
    sample_catchments = {}
    initial_len = len(uV)
    area_ID = 1
    # Iterate through nodes from upstream to downstream
    for i in np.arange(initial_len - 1, -1, -1):
        node = uV[i]  # Node in network
        if node in target_nodes:
            sample_name = sample_names[np.ravel(np.where(target_nodes == node))][0]
            unvis_up_nodes = [node]
            # Loop through unvisited nodes upstream
            for new_up_node in uV[i + 1 :]:
                # If this node drains to a node in our list we add it to the list
                # Hence we progressively go upstream building the subcatchment
                if receiver_at_node[new_up_node] in unvis_up_nodes:
                    unvis_up_nodes.append(new_up_node)
                # When we find a node not in the subcatchment we stop as we have reached a drainage divide
                else:
                    break
            sample_catchments[area_ID] = {
                "SampleName": sample_name,
                "catchment_nodes": unvis_up_nodes,
                "node": node,
            }  # Append catchment to dictionary
            uV = fast_delete(uV, unvis_up_nodes)  # Remove visited nodes from array
            area_ID += 1  # Update new node ID
    return sample_catchments


def get_subcatchment_map(model_grid, sample_catchment_dict):
    """Generates a map (2D numpy array) of the identified sub-catchments
    args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.process_topo

        sample_catchment_dict (dict): Dictionary defining sub-catchments
        (see `get_subcatchments`)

    Returns:
        out (NxM int array): Map of the areaIDs for each sub-catchment.
        See output from `save_outputs` and `get_subcatchments` for how these relate
        to the sample sites.
    """

    out = np.zeros(model_grid.shape).flatten()
    for label, catchment_dict in sample_catchment_dict.items():
        out[catchment_dict["catchment_nodes"]] = label
    out = out.reshape(model_grid.shape)
    return out


def load_sample_data(path_to_file):
    """Loads sample site data into memory:
    args:
        path_to_file (str): Path to file which contains the coordinates of
        sample sites with corresponding sample names.
        This file must have format:

        SampleName | x coordinate | y coordinate | ... 
        -----------------------------------------| ...
           string  |    float     |     float    | ..."""

    data = pd.read_csv(path_to_file).to_numpy()
    sample_sites = data[:, 1:3].astype(float)
    sample_names = data[:, 0].astype(str)
    return sample_names, sample_sites


def node_to_coord(model_grid, node):
    """Converts node IDs into true coordinates
    args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.process_topo

        node (int)): corresponding node IDs for each coordinate pair

    returns:
        coordinates (tuple): Coordinates (y,x) for the input node ID"""
    y, x = np.unravel_index(node, model_grid.shape)
    coordindates = y * model_grid.dy, x * model_grid.dx
    return coordindates


def save_subcatchments(model_grid, sample_catchment_dict, catchment_map, output_dir):
    """Save outputs from sample-processing to disk in appropriate format

    args:
        model_grid (RasterModelGrid): Model-grid with flowrouted across it
        (see auto_catchments.process_topo

        sample_catchment_dict (dict): Dictionary defining sub-catchments
        (see `get_subcatchments`)

        catchment_map (NxM int array): Map of the areaIDs for each sub-catchment.
        See output from `save_outputs` and `get_subcatchments` for how these relate
        to the sample sites.

    Returns:
        None:

    Generates two output files.
    1. "fitted_localities.csv" is a table of the coordinates of the *fitted* sample sites,
        alongside the associated sample name, and the AreaID of the sub-catchment. This areaID
        can be used to interpret/interoperate with the 'area_IDs.asc' output.
    2. "area_IDs.asc" is a map of the delineated sub-catchments. The area_IDs map onto the
        sample sites given in "fitted_localities.csv". This is an ESRI ASCII raster file format
        appropriate for use in most GIS software.
    """

    node_IDs = []
    sample_names = []
    for label, catchment_dict in sample_catchment_dict.items():
        node_IDs.append(label)  # node ID
        sample_names.append(catchment_dict["SampleName"])

    outdf = pd.DataFrame({"Catchment ID": node_IDs, "SampleName": sample_names})
    outdf.to_csv(output_dir + "/subcatch_area_IDs.csv", index=False)
    if os.path.exists(output_dir + "/area_IDs.asc"):  # Allows over-writing of .asc files
        os.remove(output_dir + "/area_IDs.asc")
    _ = model_grid.add_field("area_IDs", np.flipud(catchment_map))
    model_grid.save(output_dir + "/area_IDs.asc", names="area_IDs")
