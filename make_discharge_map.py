import autocatchments as ac
from datetime import datetime
import matplotlib.pyplot as plt
import sewage as swg

"""Fetches real-time discharge data from Thames Water and visualises downstream impacts in real time"""

print("### Loading in drainage map ###")
mg = ac.toolkit.load_d8("input_dir/thames_d8.nc")

# Twice as fast but requires pre-calculated node-arrays which
# are stored inefficiently as plain text in hard-drive
# (2*100Mb for .txt's vs 20MB for .nc)

# mg = ac.toolkit.load_from_node_arrays(
#     path_to_receiver_nodes="input_dir/thames_receiver_nodes.txt",
#     path_to_ordered_nodes="input_dir/thames_ordered_nodes.txt",
#     shape=mg.shape,
#     xy_of_lower_left=mg.xy_of_lower_left,
#     xy_spacing= (mg.dx,mg.dy)
# )

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
