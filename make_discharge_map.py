import autocatchments as ac
from datetime import datetime
import matplotlib.pyplot as plt
import sewage as swg
import pickle

"""Fetches real-time discharge data from Thames Water and visualises downstream impacts in real time"""

print("### Loading in drainage map ###")
# Load from Topographic grid
# mg = ac.toolkit.load_topo("input_dir/thames_elev.nc")

# Load from pickled file for speed
with open('input_dir/mg_elev.obj', 'rb') as handle:
    mg = pickle.load(handle)

print("### Getting sewage discharge data ###")
sewage_df = swg.get_current_discharge_status()
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string_file = now.strftime("%y%m%d_%H%M%S")

x, y, z = swg.calc_downstream_polluted_nodes(sewage_df, mg)
if len(x)>0:
    long, lat = swg.BNG_to_WGS84_points(x, y)
else:
    print("! No discharges currently occurring !")
    long, lat = [], []


print("### Plotting outputs ###")
swg.plot_sewage_map(downstream_xyz=(x, y, z), grid=mg, sewage_df=sewage_df, title=dt_string)
plt.savefig("output_dir/plots/" + dt_string_file + ".png")
plt.show()
print("### Saving outputs ###")
out_geojson = swg.xyz_to_geojson(long, lat, z, label="upstream_sources")
swg.save_json(out_geojson, "output_dir/geojsons/" + dt_string_file + ".geojson")
