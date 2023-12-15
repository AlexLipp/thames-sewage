import boto3
import json
from osgeo import ogr, osr


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

        print(
            f"All objects in '\033[92m{folder_name}\033[0m' folder deleted successfully."
        )
    else:
        print(f"No objects found in '\033[92m{folder_name}\033[0m' folder.")


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
        # Write print(f"File {file_path} uploaded to {bucket_name}/{object_name} successfully."), but make the file paths and destinations in red using ANSI escape codes
        print(
            f"File \033[92m{file_path}\033[0m uploaded to \033[92m{bucket_name}/{object_name}\033[0m successfully."
        )

    except Exception as e:
        print(f"Error uploading file: {str(e)}")


def write_timestamp(datetime_string: str, timestamp_filename: str):
    """Writes a file that contains "datetime_string" to file "timestamp_filename"""
    try:
        with open(timestamp_filename, "w") as file:
            file.write(datetime_string)
        print(
            f"Successfully wrote timestamp \033[92m{datetime_string}\033[0m to \033[92m{timestamp_filename}\033[0m."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


def project_geojson_BNG_WGS84(geojson: dict) -> dict:
    """Projects a geojson from BNG to WGS84. Modifies in place"""
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(27700)  # British National Grid
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)  # WGS84
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    for feature in geojson["features"]:
        geom = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
        geom.Transform(transform)
        feature["geometry"] = json.loads(geom.ExportToJson())
