import pickle
from datetime import datetime

import matplotlib.pyplot as plt
from autocatchments import channel_profiler as profiler

import sewage as swg

print("### Loading in drainage map ###")
# Load from Topographic grid - This is very slow (5 minutes) due to the sink filling calculation.
# Recommend loading from the pickled pre-calculated input instead...
# mg = ac.toolkit.load_topo("input_dir/thames_elev_masked.nc")

# Load the model grid from a pickled file for speed
with open("input_dir/mg_elev.obj", "rb") as handle:
    mg = pickle.load(handle)

print("### Getting sewage discharge data ###")
sewage_df = swg.get_current_discharge_status()

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string_file = now.strftime("%y%m%d_%H%M%S")
x, y, z = swg.calc_downstream_polluted_nodes(sewage_df, mg)

if swg.get_active_rows(sewage_df).size != 0:
    print("Building channel profile structure")
    cp = profiler.ChannelProfiler(
        mg,
        "number_upstream_discharges",
        minimum_channel_threshold=0.9,
        minimum_outlet_threshold=0.9,
    )
    cp.run_one_step()
    out_geojson = swg.profiler_data_struct_to_geojson(
        cp.data_structure, mg, "number_upstream_discharges"
    )

else:
    print("! No discharges currently occurring !")
    out_geojson = swg.empty_linestring_featurecollection("number_upstream_discharges")

print("### Saving outputs ###")
swg.save_json(out_geojson, "output_dir/geojsons/" + dt_string_file + ".geojson")
print("### Plotting outputs ###")
swg.plot_sewage_map(downstream_xyz=(x, y, z), grid=mg, sewage_df=sewage_df, title=dt_string)
plt.savefig("output_dir/plots/" + dt_string_file + ".png")
plt.show()
