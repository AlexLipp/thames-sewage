#!/usr/bin/python
from aux_funcs import empty_s3_folder, upload_file_to_s3, write_timestamp
from poopy.companies import ThamesWater
from datetime import datetime
import os

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

# AWS directory to save current outputs to
AWS_NOW_DIR = "now/"

# AWS directory to save long-term outputs to
AWS_PAST_DIR = "past/"

# Name of the timestamp file to upload (locally + in AWS)
TIMESTAMP_FILENAME = "timestamp.txt"

# Name of the geojson file in the AWS bucket for current discharges
AWS_GEOJSON_FILENAME = "now.geojson"


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
    _, _, _ = tw.calculate_downstream_points(include_recent_discharges=True)
    tw.save_downstream_geojson(LOCAL_GEOJSON_DIR + geojson_file_name)
    print("### Uploading outputs to AWS bucket ###")
    upload_downstream_impact_files_to_s3(
        geojson_file_name, now.isoformat(timespec="seconds")
    )
    print("Finished @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()
