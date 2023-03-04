rgn=-R320000/620000/80000/280000
# gmt grdcut ~/Documents/large_data/UK_hydro_map/uk_d8.nc $rgn -Gthames_d8.nc
gmt grdcut ~/Documents/large_data/UK_hydro_map/ihdtm-2016_4881828/14jan16_HGHT_0_000_000_700_1300.asc $rgn -Gthames_elev.nc
