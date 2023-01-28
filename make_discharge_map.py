import autocatchments as ac
from datetime import datetime
import sewage as swg

"""Fetches real-time discharge data from Thames Water and visualises downstream impacts in real time"""

print("### Loading in drainage map ###")
mg = ac.toolkit.load_d8("input_dir/thames_d8.nc")

print("### Getting sewage discharge data ###")
sewage_df = swg.get_thames_data()
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
x, y, z = swg.calc_downstream_polluted_nodes(sewage_df, mg)
swg.plot_sewage_map(downstream_xyz=(x, y, z), grid=mg, sewage_df=sewage_df, title=dt_string)
long, lat = swg.BNG_to_WGS84_points(x, y)
print("### Plotting outputs ###")
print("### Saving outputs ###")
out_geojson = swg.xyz_to_geojson(long, lat, z, label="upstream_sources")
swg.save_geojson(out_geojson, "upstream_sources.geojson")
