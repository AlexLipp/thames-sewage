rgn=-R320000/620000/80000/280000
# gmt grdcut ~/Documents/large_data/UK_hydro_map/uk_d8.nc $rgn -Gthames_d8.nc
echo "Extracting subset"
# gmt grdcut ~/Documents/large_data/UK_hydro_map/ihdtm-2016_4881828/14jan16_HGHT_0_000_000_700_1300.asc $rgn -Gthames_elev.nc
echo "Projecting onto long lats"
gdalwarp thames_elev.nc -s_srs EPSG:27700 -t_srs EPSG:4326 thames_elev_ll.nc

rgn_ll=$(gmt grdinfo thames_elev_ll.nc -Ir)
ll_space=$(gmt grdinfo thames_elev_ll.nc -Cn -o7)

echo "Calculating landmask"
# Generate a land/ocean mask for the DEM
gmt grdlandmask $rgn_ll -I$ll_space -N0/1 -Gland_mask.grd -Dh -rp
echo "Masking topo"
gmt grdmath thames_elev_ll.nc land_mask.grd MUL = thames_elev_ll_masked.nc
gdalwarp thames_elev_ll_masked.nc -s_srs EPSG:4326 -t_srs EPSG:27700 thames_elev_masked.nc

rm land_mask.grd, thames_elev_ll.nc,thames_elev_ll_masked.nc
