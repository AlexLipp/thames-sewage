import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from landlab.components.flow_accum.flow_accum_bw import find_drainage_area_and_discharge
from landlab import RasterModelGrid
from typing import Tuple
import json
from geojson import FeatureCollection, Point, Feature


def get_thames_data():
    """Gets real-time data from the Thames Water API"""
    # add in your API credentials here
    clientID = "8a10d9580e9b4a0db6f1b2ae7ee19f7c"
    clientSecret = "FD8A75e4e5a84aB19Cf9abDfAebC31eA"

    # modify this url as desired to access the different end points. e.g. replace DischargeCurrentStatus at the end of the resource URL
    api_root = "https://prod-tw-opendata-app.uk-e1.cloudhub.io"
    api_resource = "/data/STE/v1/DischargeCurrentStatus"
    url = api_root + api_resource

    # add here any query parameters if using them e.g. date filters, leave as '' for none.
    params = ""

    # send the request
    r = requests.get(
        url, headers={"client_id": clientID, "client_secret": clientSecret}, params=params
    )
    print("Requesting from " + r.url)

    # check response status and use only valid requests
    if r.status_code == 200:
        response = r.json()
        df = pd.json_normalize(response, "items")
    else:
        raise Exception(
            "Request failed with status code {0}, and error message: {1}".format(
                r.status_code, r.json()
            )
        )

    # you can then manipulate the dataframe df as you wish:
    return df


def geographic_coords_to_model_xy(xy_coords, grid):
    """Converts geographical coordinates (from lower left) into model
    grid coordinates (from upper left)"""
    xy_of_upper_left = grid.xy_of_lower_left[0], grid.xy_of_lower_left[1] + grid.dy * grid.shape[0]
    x = (xy_coords[0] - xy_of_upper_left[0]) / grid.dx
    y = (xy_of_upper_left[1] - xy_coords[1]) / grid.dy
    return x, y


def model_xy_to_geographic_coords(model_xy_coords, grid):
    """Converts model grid coordinates (from upper left) to geographical coordinates
    (from lower left)"""
    xy_of_upper_left = grid.xy_of_lower_left[0], grid.xy_of_lower_left[1] + grid.dy * grid.shape[0]
    x = xy_of_upper_left[0] + model_xy_coords[0] * grid.dx
    y = xy_of_upper_left[1] - model_xy_coords[1] * grid.dy
    return x, y


def get_active_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Only returns rows actively discharging"""
    return sewage_df.loc[sewage_df["AlertStatus"] == "Discharging", :]


def get_inactive_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Returns rows not actively discharging"""
    return sewage_df.loc[sewage_df["AlertStatus"] == "Not discharging", :]


def calc_downstream_polluted_nodes(
    sewage_df: pd.DataFrame, mg: RasterModelGrid
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagates sewage sources downstream returning tuple of
    x,y (of downstream nodes) and z (number of upstream sources)"""

    active = get_active_rows(sewage_df)
    x, y = geographic_coords_to_model_xy((active["X"].to_numpy(), active["Y"].to_numpy()), mg)
    nodes = np.ravel_multi_index(
        (y.astype(int), x.astype(int)), mg.shape
    )  # Grid nodes of point sources

    source_array = np.zeros(mg.shape).flatten()
    source_array[nodes] = 1
    # Integrate downstream
    _, number_upstream_sources = find_drainage_area_and_discharge(
        mg.at_node["flow__upstream_node_order"],
        r=mg.at_node["flow__receiver_node"],
        runoff=source_array,
    )
    # Nodes downstream of a sewage source
    dstr_polluted_nodes = np.where(number_upstream_sources != 0)[0]
    # Number of upstream nodes at sites
    dstr_polluted_vals = number_upstream_sources[dstr_polluted_nodes]
    dstr_polluted_gridy, dstr_polluted_gridx = np.unravel_index(dstr_polluted_nodes, mg.shape)
    dstr_polluted_xy = model_xy_to_geographic_coords((dstr_polluted_gridx, dstr_polluted_gridy), mg)
    return (dstr_polluted_xy[0], dstr_polluted_xy[1], dstr_polluted_vals)


def plot_sewage_map(
    downstream_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: RasterModelGrid,
    sewage_df: pd.DataFrame,
    title: str,
) -> None:
    """Visualises downstream pollution"""
    x = downstream_xyz[0]
    y = downstream_xyz[1]
    z = downstream_xyz[2]

    active = get_active_rows(sewage_df)
    inactive = get_inactive_rows(sewage_df)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    area = grid.at_node["drainage_area"]
    plt.imshow(
        area.reshape(grid.shape),
        norm=LogNorm(),
        cmap="Greys",
        extent=[
            grid.xy_of_lower_left[0],
            grid.xy_of_lower_left[0] + grid.shape[1] * grid.dx,
            grid.xy_of_lower_left[1],
            grid.xy_of_lower_left[1] + grid.shape[0] * grid.dy,
        ],
    )
    cb = plt.colorbar()
    cb.set_label("Drainage Area m$^2$")
    plt.xlabel("Easting")
    plt.ylabel("Northing")

    plt.scatter(inactive["X"], inactive["Y"], c="green", marker="x", s=10)
    plt.scatter(active["X"], active["Y"], c="red", marker="x")
    plt.legend(["Inactive", "Discharging"])
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.imshow(
        area.reshape(grid.shape),
        norm=LogNorm(),
        cmap="Greys",
        extent=[
            grid.xy_of_lower_left[0],
            grid.xy_of_lower_left[0] + grid.shape[1] * grid.dx,
            grid.xy_of_lower_left[1],
            grid.xy_of_lower_left[1] + grid.shape[0] * grid.dy,
        ],
    )
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.scatter(x=x, y=y, c=z, s=0.2, cmap="plasma")
    cb = plt.colorbar()
    cb.set_label("# Upstream Discharges")
    plt.scatter(active["X"], active["Y"], c="red", marker="x")
    plt.title(title)
    plt.show()


def xyz_to_geojson(
    xs: np.ndarray, ys: np.ndarray, vals: np.ndarray, label: str
) -> FeatureCollection:
    """Converts sequence of x, y and 'values' into a geojson FeatureCollection of points.
    `label' indicates the key of the properties dataframe to store vals"""
    n_points = len(xs)
    geojson_points = FeatureCollection(
        [
            Feature(geometry=Point((xs[i], ys[i])), properties={label: vals[i]})
            for i in range(n_points)
        ]
    )
    return geojson_points


def save_geojson(object, filename: str) -> None:
    """Saves a geojson object to file"""
    f = open(filename, "w")
    json.dump(object, f)
