"""
Auxiliary functions for uploading files to a AWS S3 bucket
"""

import boto3
import os
from datetime import datetime

BUCKET_NAME = "thamessewage"
PROFILE_NAME = os.getenv("S3_PROFILE_NAME")


def empty_s3_folder(bucket_name: str, folder_name: str, profile_name: str) -> None:
    """Empties a folder in an AWS bucket"""

    session = boto3.Session(profile_name=profile_name)
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


def upload_file_to_s3(
    file_path: str, bucket_name: str, object_name: str, profile_name: str
):
    """Uploads a file to an AWS bucket"""
    session = boto3.Session(profile_name=profile_name)
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


def write_timestamp(datetime_string: str):
    """Writes a file called "timestamp.txt" to file that contains a datetime"""
    try:
        with open("output_dir/timestamp.txt", "w") as file:
            file.write(datetime_string)
        print("Successfully created and wrote to 'timestamp.txt'")
    except Exception as e:
        print(f"An error occurred: {e}")


def upload_downstream_impact_files_to_s3(file_path: str) -> None:
    """Uploads geojson files to ThamesSewage AWS bucket"""

    empty_s3_folder(
        bucket_name=BUCKET_NAME, folder_name="now/", profile_name=PROFILE_NAME
    )  # Empty the 'now' folder
    # Upload file to current 'now' output and also the long-term storage 'past' folder
    upload_file_to_s3(
        file_path="output_dir/geojsons/" + file_path,
        bucket_name=BUCKET_NAME,
        object_name="now/now.geojson",
        profile_name=PROFILE_NAME,
    )
    upload_file_to_s3(
        file_path="output_dir/geojsons/" + file_path,
        bucket_name=BUCKET_NAME,
        object_name="past/" + file_path,
        profile_name=PROFILE_NAME,
    )
    # Add timestamp file to now folder

    write_timestamp(datetime.now().isoformat(timespec="seconds"))

    upload_file_to_s3(
        file_path="output_dir/timestamp.txt",
        bucket_name=BUCKET_NAME,
        object_name="now/timestamp.txt",
        profile_name=PROFILE_NAME,
    )
