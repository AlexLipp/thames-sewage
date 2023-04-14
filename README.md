# thames-sewage
Realtime mapping of sewage release events in Thames basin

## Installation 

In addition to the dependencies listed in `requirements.txt`, this script depends on the custom-made package [`autocatchments`](https://github.com/AlexLipp/autocatchments) which was created initially for personal use only. 

Install miniconda
Install everything via conda
may need to symlink libtiff
do conda update --all at the end


conda create -n thames-sewage python=3.10

conda install -c conda-forge matplotlib geojson landlab gdal
conda install -c anaconda numpy pandas requests
pip install git+https://github.com/AlexLipp/autocatchments.git


todo: make output_dirs/plots & output_dirs/geojsons folder dynamically, or commit the empty dirs


## Usage 

Running `python make_discharge_map.py` from the command line will generate a time-stamped map of the number of upstream sewage discharge events at any point in the Thames basin. An example of this output is shown below. It additionally outputs the results as a `geoJSON` which can be loaded in to any standard GIS environment.
![28123_2032](https://user-images.githubusercontent.com/10188895/215289603-3315e7b6-5a50-48ed-9ef0-7a9269e5e2e3.png)






mkdir -p ~/.aws-lambda-rie
curl -Lo ~/.aws-lambda-rie/aws-lambda-rie https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie
chmod +x ~/.aws-lambda-rie/aws-lambda-rie


docker build -t sewage .

docker run -v ~/.aws-lambda-rie:/aws-lambda -p 9000:8080 --platform=linux/amd64 --entrypoint /aws-lambda/aws-lambda-rie sewage /opt/conda-env/bin/python -m awslambdaric sewage.make_discharge_map

then:

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"payload":"hello world!"}'