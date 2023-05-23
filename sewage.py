import datetime
import json
import pickle
from datetime import datetime
from typing import Tuple

import autocatchments as ac
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from autocatchments import channel_profiler as profiler
from geojson import Feature, FeatureCollection, LineString, Point
from landlab import RasterModelGrid
from landlab.components.flow_accum.flow_accum_bw import find_drainage_area_and_discharge
from matplotlib.colors import LogNorm
from osgeo import osr


def get_all_discharge_starts():
    """Gets all discharge starts that have occurred since API online in April 2022"""
    # add in your API credentials here
    clientID = "8a10d9580e9b4a0db6f1b2ae7ee19f7c"
    clientSecret = "FD8A75e4e5a84aB19Cf9abDfAebC31eA"

    api_root = "https://prod-tw-opendata-app.uk-e1.cloudhub.io"
    api_resource = "/data/STE/v1/DischargeAlerts"
    url = api_root + api_resource

    # Iterate through using the 1000 output limit
    num_outputs = 1000
    i = 0
    df = pd.DataFrame()
    while num_outputs == 1000:
        # Only extract the items corresponding to discharge *starts*.
        params = {
            "limit": 1000,
            "offset": i * 1000,
            "col_1": "AlertType",
            "operand_1": "eq",
            "value_1": "Start",
        }

        # send the request
        r = requests.get(
            url,
            headers={"client_id": clientID, "client_secret": clientSecret},
            params=params,
        )
        print("Requesting from " + r.url)

        # check response status and use only valid requests
        if r.status_code == 200:
            response = r.json()
            df_temp = pd.json_normalize(response, "items")
        else:
            raise Exception(
                "Request failed with status code {0}, and error message: {1}".format(
                    r.status_code, r.json()
                )
            )
        df = pd.concat([df, df_temp])
        i += 1
        num_outputs = df_temp.shape[0]
    print("Returning", df.shape[0], "`Start` records")
    return df


def minutes_elapsed(iso_str1, iso_str2):
    """Calculates minutes elapsed between two ISO8601 strings"""
    dt1 = datetime.fromisoformat(iso_str1)
    dt2 = datetime.fromisoformat(iso_str2)

    if dt2 < dt1:  # swap if necessary so that dt1 is earlier
        dt1, dt2 = dt2, dt1

    elapsed_days = (dt2 - dt1).days
    elapsed_minutes = (elapsed_days * 24 * 60) + ((dt2 - dt1).seconds // 60)

    return elapsed_minutes


def get_all_discharge_alerts():
    """Gets all discharge alerts (e.g., Start, Stop and offline/online status) that have occurred since API online in April 2022"""
    # add in your API credentials here
    clientID = "8a10d9580e9b4a0db6f1b2ae7ee19f7c"
    clientSecret = "FD8A75e4e5a84aB19Cf9abDfAebC31eA"

    api_root = "https://prod-tw-opendata-app.uk-e1.cloudhub.io"
    api_resource = "/data/STE/v1/DischargeAlerts"
    url = api_root + api_resource

    # Iterate through using the 1000 output limit
    num_outputs = 1000
    i = 0
    df = pd.DataFrame()
    while num_outputs == 1000:
        # Only extract the items corresponding to discharge *starts*.
        params = {"limit": 1000, "offset": i * 1000}

        # send the request
        r = requests.get(
            url,
            headers={"client_id": clientID, "client_secret": clientSecret},
            params=params,
        )
        print("Requesting from " + r.url)

        # check response status and use only valid requests
        if r.status_code == 200:
            response = r.json()
            df_temp = pd.json_normalize(response, "items")
        else:
            raise Exception(
                "Request failed with status code {0}, and error message: {1}".format(
                    r.status_code, r.json()
                )
            )
        df = pd.concat([df, df_temp])
        i += 1
        num_outputs = df_temp.shape[0]
    print("Returning", df.shape[0], "records")
    return df


def get_all_discharge_stops():
    """Gets all discharge stops that have occurred since API online in April 2022"""
    # add in your API credentials here
    clientID = "8a10d9580e9b4a0db6f1b2ae7ee19f7c"
    clientSecret = "FD8A75e4e5a84aB19Cf9abDfAebC31eA"

    api_root = "https://prod-tw-opendata-app.uk-e1.cloudhub.io"
    api_resource = "/data/STE/v1/DischargeAlerts"
    url = api_root + api_resource

    # Iterate through using the 1000 output limit
    num_outputs = 1000
    i = 0
    df = pd.DataFrame()
    while num_outputs == 1000:
        # Only extract the items corresponding to discharge *starts*.
        params = {
            "limit": 1000,
            "offset": i * 1000,
            "col_1": "AlertType",
            "operand_1": "eq",
            "value_1": "Stop",
        }

        # send the request
        r = requests.get(
            url,
            headers={"client_id": clientID, "client_secret": clientSecret},
            params=params,
        )
        print("Requesting from " + r.url)

        # check response status and use only valid requests
        if r.status_code == 200:
            response = r.json()
            df_temp = pd.json_normalize(response, "items")
        else:
            raise Exception(
                "Request failed with status code {0}, and error message: {1}".format(
                    r.status_code, r.json()
                )
            )
        df = pd.concat([df, df_temp])
        i += 1
        num_outputs = df_temp.shape[0]
    print("Returning", df.shape[0], "`Stop` records")
    return df


def get_current_discharge_status():
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
        url,
        headers={"client_id": clientID, "client_secret": clientSecret},
        params=params,
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

    if df.shape[0] == 1000:
        raise Exception(
            "Warning: Number of outputs is at or exceeds 1000 output limit. \nOutputs may be incomplete"
        )
    print("Returning", df.shape[0], "records")
    return df


def geographic_coords_to_model_xy(xy_coords, grid):
    """Converts geographical coordinates (from lower left) into model
    grid coordinates (from upper left)"""
    xy_of_upper_left = (
        grid.xy_of_lower_left[0],
        grid.xy_of_lower_left[1] + grid.dy * grid.shape[0],
    )
    x = (xy_coords[0] - xy_of_upper_left[0]) / grid.dx
    y = (xy_of_upper_left[1] - xy_coords[1]) / grid.dy
    return x, y


def model_xy_to_geographic_coords(model_xy_coords, grid):
    """Converts model grid coordinates (from upper left) to geographical coordinates
    (from lower left)"""
    xy_of_upper_left = (
        grid.xy_of_lower_left[0],
        grid.xy_of_lower_left[1] + grid.dy * grid.shape[0],
    )
    x = xy_of_upper_left[0] + model_xy_coords[0] * grid.dx
    y = xy_of_upper_left[1] - model_xy_coords[1] * grid.dy
    return x, y


def get_active_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Only returns rows actively discharging"""
    return sewage_df.loc[sewage_df["AlertStatus"] == "Discharging", :]


def get_recent_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Only returns rows actively discharging"""
    return sewage_df.loc[sewage_df["AlertPast48Hours"], :]


def get_active_and_recent_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Returns rows actively discharging or that discharged in last 48 hours"""
    return sewage_df.loc[
        np.logical_or(
            sewage_df["AlertStatus"] == "Discharging", sewage_df["AlertPast48Hours"]
        ),
        :,
    ]


def get_inactive_rows(sewage_df: pd.DataFrame) -> pd.DataFrame:
    """Returns rows not actively discharging"""
    return sewage_df.loc[sewage_df["AlertStatus"] == "Not discharging", :]


def calc_downstream_polluted_nodes(
    sewage_df: pd.DataFrame, mg: RasterModelGrid
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagates sewage sources downstream returning tuple of
    x,y (of downstream nodes) and z (number of upstream sources)"""

    # active = get_active_rows(sewage_df)
    active = get_active_and_recent_rows(sewage_df)

    x, y = geographic_coords_to_model_xy(
        (active["X"].to_numpy(), active["Y"].to_numpy()), mg
    )
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
    # Append number of upstream sources as a field to the RasterModelGrid
    mg.add_field("number_upstream_discharges", number_upstream_sources)
    # Nodes downstream of a sewage source
    dstr_polluted_nodes = np.where(number_upstream_sources != 0)[0]
    # Number of upstream nodes at sites
    dstr_polluted_vals = number_upstream_sources[dstr_polluted_nodes]
    dstr_polluted_gridy, dstr_polluted_gridx = np.unravel_index(
        dstr_polluted_nodes, mg.shape
    )
    dstr_polluted_xy = model_xy_to_geographic_coords(
        (dstr_polluted_gridx, dstr_polluted_gridy), mg
    )
    return (dstr_polluted_xy[0], dstr_polluted_xy[1], dstr_polluted_vals)


def alerts_to_events_df(alerts: pd.DataFrame) -> pd.DataFrame:
    """Converts dataframe of all discharge outlet alerts and returns a
    dataframe of specfic discharge events with the following columns:
    'LocationName, PermitNumber, LocationGridRef, X, Y, ReceivingWaterCourse, StartTime,
     StopTime, Duration , CompleteOutput, OngoingDischarge'"""
    events = []
    for site in alerts["PermitNumber"].unique():
        # Iterate over all sites
        site_alerts = alerts.loc[alerts["PermitNumber"] == site]
        n_alerts = site_alerts.shape[0]
        for start_index in np.where(site_alerts["AlertType"] == "Start")[0]:
            start_alert = site_alerts.iloc[start_index]
            discharge_event = {
                "LocationName": start_alert["LocationName"],
                "PermitNumber": start_alert["PermitNumber"],
                "LocationGridRef": start_alert["LocationGridRef"],
                "X": start_alert["X"],
                "Y": start_alert["Y"],
                "ReceivingWaterCourse": start_alert["ReceivingWaterCourse"],
                "StartDateTime": start_alert["DateTime"],
                "StopDateTime": None,
                "Duration": None,
                "CompleteOutput": None,
                "OngoingDischarge": None,
            }

            # Iterate over every "Start" alert for that site
            if start_index == n_alerts - 1:
                # If start_index is last row, then station is currently discharging
                # print("Ongoing discharge")
                # print(start_alert)
                discharge_event["OngoingDischarge"] = True
                now = datetime.now().isoformat(timespec="seconds")
                discharge_event["StopDateTime"] = now
            else:
                # Fetch the next alert from that station after discharge started
                next_alert = site_alerts.iloc[start_index + 1]
                discharge_event["OngoingDischarge"] = False
                discharge_event["StopDateTime"] = next_alert["DateTime"]
                if next_alert["AlertType"] == "Stop":
                    # Discharge 'start' is followed by a 'stop'
                    # print("Discharge start followed by a stop")
                    discharge_event["CompleteOutput"] = True
                elif next_alert["AlertType"] == "Offline start":
                    # Discharge 'start' followed by the station going offline (bad!)
                    # print("Discharge info stopped by offline status")
                    discharge_event["CompleteOutput"] = False

                elif next_alert["AlertType"] == "Offline stop":
                    # This should never happen, so we raise an exception.
                    raise Exception(
                        "Error: AlertType `Start` followed by `Offline Start`"
                        + "\n"
                        + "Offending discharge:"
                        + "\n"
                        + str(site_alerts.iloc[start_index])
                    )
            discharge_event["Duration"] = minutes_elapsed(
                discharge_event["StartDateTime"], discharge_event["StopDateTime"]
            )
            events += [discharge_event]
    out_df = pd.DataFrame(events)
    out_df.sort_values(by="StartDateTime", inplace=True)
    out_df.reset_index(inplace=True, drop=True)
    return out_df


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
    recent = get_recent_rows(sewage_df)
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
    plt.scatter(recent["X"], recent["Y"], c="orange", marker="x", s=10)
    plt.legend(["Inactive", "Discharging", "Active in last 48 hrs"])
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
    plt.scatter(recent["X"], recent["Y"], c="orange", marker="x", s=10)
    plt.title(title)


def xyz_to_geojson_points(
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


def save_json(object, filename: str) -> None:
    """Saves a (geo)json object to file"""
    f = open(filename, "w")
    json.dump(object, f)


def BNG_to_WGS84_points(
    eastings: np.ndarray, northings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts coorindates on British National Grid into Long, Lat on WGS84"""

    OSR_WGS84_REF = osr.SpatialReference()
    OSR_WGS84_REF.ImportFromEPSG(4326)

    OSR_BNG_REF = osr.SpatialReference()
    OSR_BNG_REF.ImportFromEPSG(27700)

    OSR_BNG_to_WGS84 = osr.CoordinateTransformation(OSR_BNG_REF, OSR_WGS84_REF)
    lat_long_tuple_list = OSR_BNG_to_WGS84.TransformPoints(
        np.vstack([eastings, northings]).T
    )
    lat_long_array = np.array(list(map(np.array, lat_long_tuple_list)))
    return (lat_long_array[:, 1], lat_long_array[:, 0])


def ids_to_xyz(
    node_ids: np.ndarray, grid: RasterModelGrid, field: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts list of node ids into arrays of model x, y coordinates
    and values of a given field"""

    model_ys, model_xs = np.unravel_index(node_ids, grid.shape)
    xs, ys = ac.toolkit.model_xy_to_geographic_coords((model_xs, model_ys), grid)
    vals = grid.at_node[field][node_ids]
    return (xs, ys, vals)


def xyz_to_linestring(xs: np.ndarray, ys: np.ndarray, label: str, value: float):
    """Turns a list of x,y coordinates and a given label, value pair into
    a geojson LineString feature"""
    geom = LineString(coordinates=tuple(zip(xs, ys)))
    prop = {label: value}
    return Feature(geometry=geom, properties=prop)


def profiler_data_struct_to_geojson(
    profiler_data_struct, grid: RasterModelGrid, field: str
) -> FeatureCollection:
    """Turns output from ChannelProfiler into a geojson FeatureCollection
    of LineStrings with property corresponding to chosen field"""
    features = []
    for _, segments in profiler_data_struct.items():
        for _, segment in segments.items():
            xs, ys, vals = ids_to_xyz(segment["ids"], grid, field)
            longs, lats = BNG_to_WGS84_points(xs, ys)
            features += [xyz_to_linestring(longs, lats, field, vals[-1])]

    return FeatureCollection(features)


def empty_linestring_featurecollection(label: str):
    """Generates an empty linestring geojson feature collection"""
    feat = Feature(geometry=LineString(([])), properties={label: 0})
    return FeatureCollection([feat])


def upload_file_to_s3(file_path: str, bucket_name: str, object_name: str):
    """Uploads a file to an AWS bucket"""
    session = boto3.Session(profile_name="alex")
    s3 = session.client("s3")
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        # Give public read access
        s3.put_object_acl(ACL="public-read", Bucket=bucket_name, Key=object_name)
        # Set cache-control headers to prevent caching
        s3.put_object_tagging(
            Bucket=bucket_name,
            Key=object_name,
            Tagging={"TagSet": [{"Key": "Cache-Control", "Value": "no-cache"}]},
        )
        print("File uploaded successfully.")
    except Exception as e:
        print(f"Error uploading file: {str(e)}")


def empty_s3_folder(bucket_name: str, folder_name: str) -> None:
    """Empties a folder in an AWS bucket"""

    session = boto3.Session(profile_name="alex")
    s3 = session.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    if "Contents" in response:
        # Construct the list of objects to delete
        objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

        # Perform the batch delete operation
        s3.delete_objects(Bucket=bucket_name, Delete={"Objects": objects_to_delete})

        print(f"All objects in '{folder_name}' folder deleted successfully.")
    else:
        print(f"No objects found in '{folder_name}' folder.")


def make_discharge_map():
    print("### Loading in drainage map ###")
    # Load from Topographic grid - This is very slow (5 minutes) due to the sink filling calculation.
    # Recommend loading from the pickled pre-calculated input instead...
    # mg = ac.toolkit.load_topo("input_dir/thames_elev_masked.nc")

    # Load the model grid from a pickled file for speed
    with open("input_dir/mg_elev.obj", "rb") as handle:
        mg = pickle.load(handle)

    print("### Getting sewage discharge data ###")
    sewage_df = get_current_discharge_status()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_file = now.strftime("%y%m%d_%H%M%S")
    x, y, z = calc_downstream_polluted_nodes(sewage_df, mg)

    if get_active_and_recent_rows(sewage_df).shape[0] != 0:
        print(
            get_active_and_recent_rows(sewage_df).shape[0],
            " CSOs discharging now or within the last 48 hours",
        )
        print("Building channel profile structure")
        cp = profiler.ChannelProfiler(
            mg,
            "number_upstream_discharges",
            minimum_channel_threshold=0.9,
            minimum_outlet_threshold=0.9,
        )
        cp.run_one_step()
        out_geojson = profiler_data_struct_to_geojson(
            cp.data_structure, mg, "number_upstream_discharges"
        )

    else:
        print("! No discharges currently occurring !")
        out_geojson = empty_linestring_featurecollection("number_upstream_discharges")

    print("### Saving outputs ###")
    save_json(out_geojson, "output_dir/geojsons/" + dt_string_file + ".geojson")

    print("### Uploading outputs to AWS bucket ###")
    file_path = "output_dir/geojsons/" + dt_string_file + ".geojson"
    bucket_name = "thamessewage"  # S3 bucket name
    aws_object_name = (
        dt_string_file + ".geojson"
    )  # The name of the file in the S3 bucket

    empty_s3_folder(
        bucket_name=bucket_name, folder_name="now/"
    )  # Empty the 'now' folder
    # Upload file to current 'now' output and also the long-term storage 'past' folder
    upload_file_to_s3(
        file_path=file_path, bucket_name=bucket_name, object_name="now/now.geojson"
    )
    upload_file_to_s3(
        file_path=file_path,
        bucket_name=bucket_name,
        object_name="past/" + aws_object_name,
    )

    print("### Plotting outputs ###")
    plot_sewage_map(
        downstream_xyz=(x, y, z), grid=mg, sewage_df=sewage_df, title=dt_string
    )
    plt.savefig("output_dir/plots/" + dt_string_file + ".png")


def get_all_past_discharges():
    """Retrieves all discharge events that have occurred/are occurring
    since start of data records. Saves as a .json and upload to AWS bucket"""

    # Set up
    now = datetime.now()
    dt_string_file = now.strftime("%y%m%d_%H%M%S")
    bucket_name = "thamessewage"  # S3 bucket name
    aws_object_name = dt_string_file + ".json"  # The name of the file in the S3 bucket
    file_path = "output_dir/discharges_to_date/discharges.json"

    alerts = get_all_discharge_alerts()
    alerts.sort_values(by="DateTime", ascending=True, inplace=True)
    events_df = alerts_to_events_df(alerts)
    events_dict = events_df.to_dict()
    save_json(events_dict, file_path)

    empty_s3_folder(
        bucket_name=bucket_name, folder_name="discharges_to_date/"
    )  # Empty the 'now' folder
    upload_file_to_s3(
        file_path=file_path,
        bucket_name=bucket_name,
        object_name="discharges_to_date/" + aws_object_name,
    )
