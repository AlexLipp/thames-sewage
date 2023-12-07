#!/usr/bin/python
from aux_funcs import upload_downstream_impact_files_to_s3
from poopy.companies import ThamesWater
from datetime import datetime
import os


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Starting @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Get the client ID and secret from the environment
    tw_clientID = os.getenv("TW_CLIENT_ID")
    tw_clientSecret = os.getenv("TW_CLIENT_SECRET")

    if tw_clientID is None or tw_clientSecret is None:
        raise ValueError(
            "Thames Water API keys are missing from the environment!\n Please set them and try again."
        )

    now = datetime.now()
    # Generate a datetimestring in the form YYMMDDHHMMSS
    geojson_file_name = now.strftime("%y%m%d_%H%M%S.geojson")
    tw = ThamesWater(tw_clientID, tw_clientSecret)
    tw.save_downstream_geojson("output_dir/geojsons/" + geojson_file_name)
    print("### Uploading outputs to AWS bucket ###")
    upload_downstream_impact_files_to_s3(geojson_file_name)

    print("Finished @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()
