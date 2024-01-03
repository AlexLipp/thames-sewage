#!/usr/bin/python
from datetime import datetime
import json
import os

from geojson import Feature, FeatureCollection

from aux_funcs import (
    empty_s3_folder,
    upload_file_to_s3,
    write_timestamp,
    project_geojson_BNG_WGS84,
)

from poopy.companies import ThamesWater

# Name of the bucket to upload to
BUCKET_NAME = "thamessewage"

# Name of the AWS profile to use (set as an environment variable)
PROFILE_NAME = os.getenv("S3_PROFILE_NAME")
if PROFILE_NAME is None:
    raise ValueError(
        "AWS profile name is missing from the environment!\n Please set it and try again."
    )

# Local directory to save outputs to
LOCAL_OUTPUT_DIR = "output_dir/"
# Local directory to save geojsons to
LOCAL_GEOJSON_DIR = LOCAL_OUTPUT_DIR + "geojsons/"
# Local directory to save historical data to
LOCAL_HISTORICAL_DATA_DIR = LOCAL_OUTPUT_DIR + "discharges_to_date/"
# AWS directory to save current outputs to
AWS_NOW_DIR = "now/"
# AWS directory to save historical outputs to
AWS_HISTORICAL_DIR = "discharges_to_date/"
# AWS directory to save long-term outputs to
AWS_PAST_DIR = "past/"
# Name of the timestamp file to upload (locally + in AWS)
TIMESTAMP_FILENAME = "timestamp.txt"
# Name of the geojson file in the AWS bucket for current discharges
AWS_GEOJSON_FILENAME = "now.geojson"
# Name of the json file in the AWS bucket for historical discharges
AWS_JSON_FILENAME = "up_to_now.json"


def upload_downstream_impact_files_to_s3(
    geojson_file_path: str, timestamp: str
) -> None:
    """Uploads the downstream impact files to the ThamesSewage AWS bucket"""
    # Empty the 'now' folder
    empty_s3_folder(
        bucket_name=BUCKET_NAME, folder_name=AWS_NOW_DIR, profile_name=PROFILE_NAME
    )
    # Upload file to current 'now' output and also the long-term storage 'past' folder
    upload_file_to_s3(
        file_path=LOCAL_GEOJSON_DIR + geojson_file_path,
        bucket_name=BUCKET_NAME,
        object_name=AWS_NOW_DIR + AWS_GEOJSON_FILENAME,
        profile_name=PROFILE_NAME,
    )
    upload_file_to_s3(
        file_path=LOCAL_GEOJSON_DIR + geojson_file_path,
        bucket_name=BUCKET_NAME,
        object_name=AWS_PAST_DIR + geojson_file_path,
        profile_name=PROFILE_NAME,
    )
    # Add timestamp file to now folder
    write_timestamp(
        datetime_string=timestamp,
        timestamp_filename=LOCAL_OUTPUT_DIR + TIMESTAMP_FILENAME,
    )
    upload_file_to_s3(
        file_path=LOCAL_OUTPUT_DIR + TIMESTAMP_FILENAME,
        bucket_name=BUCKET_NAME,
        object_name=AWS_NOW_DIR + TIMESTAMP_FILENAME,
        profile_name=PROFILE_NAME,
    )


def upload_historical_data_files_to_s3(json_file_path: str, timestamp: str) -> None:
    """Uploads the downstream impact files to the ThamesSewage AWS bucket"""
    # Empty the 'now' folder
    empty_s3_folder(
        bucket_name=BUCKET_NAME,
        folder_name=AWS_HISTORICAL_DIR,
        profile_name=PROFILE_NAME,
    )
    # Upload file to current 'now' output and also the long-term storage 'past' folder
    upload_file_to_s3(
        file_path=LOCAL_HISTORICAL_DATA_DIR + json_file_path,
        bucket_name=BUCKET_NAME,
        object_name=AWS_HISTORICAL_DIR + AWS_JSON_FILENAME,
        profile_name=PROFILE_NAME,
    )
    # Add timestamp file to now folder
    write_timestamp(
        datetime_string=timestamp,
        timestamp_filename=LOCAL_OUTPUT_DIR + TIMESTAMP_FILENAME,
    )
    upload_file_to_s3(
        file_path=LOCAL_OUTPUT_DIR + TIMESTAMP_FILENAME,
        bucket_name=BUCKET_NAME,
        object_name=AWS_HISTORICAL_DIR + TIMESTAMP_FILENAME,
        profile_name=PROFILE_NAME,
    )


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Starting @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    tw_clientID = os.getenv("TW_CLIENT_ID")
    tw_clientSecret = os.getenv("TW_CLIENT_SECRET")
    if tw_clientID is None or tw_clientSecret is None:
        raise ValueError(
            "Thames Water API keys are missing from the environment!\n Please set them and try again."
        )
    now = datetime.now()
    geojson_file_name = now.strftime("%y%m%d_%H%M%S.geojson")
    tw = ThamesWater(tw_clientID, tw_clientSecret)

    print("Calculating current downstream discharge information")
    geojson = tw.get_downstream_geojson(include_recent_discharges=True)
    geojson = project_geojson_BNG_WGS84(geojson)  # Project to WGS84 from BNG
    # Save geojson to local directory

    # For legacy reasons we need to wrap the geojson in a FeatureCollection...
    feature_coll = FeatureCollection(
        [Feature(geometry=geojson, type="MultiLineString")]
    )

    with open(LOCAL_GEOJSON_DIR + geojson_file_name, "w") as f:
        json.dump(feature_coll, f)
    print("Uploading outputs to AWS bucket")
    upload_downstream_impact_files_to_s3(
        geojson_file_name, now.isoformat(timespec="seconds")
    )

    print("Fetching historical discharge information")
    json_file_name = now.strftime("%y%m%d_%H%M%S.json")
    tw.set_all_histories()
    df = tw.history_to_discharge_df()
    # Fill in missing stop times (for ongoing discharges) with now for consistency with www.sewagemap.com legacy inputs
    df["StopDateTime"] = df["StopDateTime"].fillna(datetime.now())
    df.to_json(LOCAL_HISTORICAL_DATA_DIR + json_file_name)
    print("Uploading outputs to AWS bucket")
    upload_historical_data_files_to_s3(
        json_file_name, now.isoformat(timespec="seconds")
    )

    print("Finished @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()
