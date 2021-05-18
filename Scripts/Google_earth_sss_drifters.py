from datetime import timedelta
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import cmocean.cm as cmo
import xarray
import cf_xarray
from parcels import (FieldSet, ParticleSet, 
                     JITParticle, ParticleFile,
                     AdvectionRK4, ErrorCode)

def DeleteParticle(particle, fieldset, time):
    particle.delete()

# property name and plotting parameters for field
prop_name = 'salt'
vmin = 25
vmax = 36.5
cmap = cmo.haline
units = 'Salinity [g/kg]'
pixels = 300  # pixels of the max. dimension

# Dataset to load
url = 'https://gcoos5.geos.tamu.edu/thredds/dodsC/ROFS_latest/txla2_his_f_latest.nc'
ds = xarray.open_dataset(url)
timestamp = str(ds.ocean_time.min().values)[:13]

kml_preamble = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.1">
<Document>
  <name>TXLA surface salinity</name>
  <open>1</open>
  <Folder>
    <name>Frames</name>
'''

kml_frame = '''  <GroundOverlay>
      <TimeSpan>
        <begin>__TIMEBEGIN__</begin>
        <end>__TIMEEND__</end>
      </TimeSpan>
      <color>__COLOR__</color>
      <Icon>
        <href>__FRAME__</href>
      </Icon>
      <LatLonBox>
        <north>__NORTH__</north>
        <south>__SOUTH__</south>
        <east> __EAST__</east>
        <west> __WEST__</west>
      </LatLonBox>
  </GroundOverlay>
'''

kml_legend = '''<ScreenOverlay>
    <name>Legend</name>
    <Icon>
        <href>legend.png</href>
    </Icon>
    <overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>
    <screenXY x="0.015" y="0.075" xunits="fraction" yunits="fraction"/>
    <rotationXY x="0.5" y="0.5" xunits="fraction" yunits="fraction"/>
    <size x="0" y="0" xunits="pixels" yunits="pixels"/>
</ScreenOverlay>
'''

kml_closing = '''  </Folder>
</Document>
</kml>
'''

name = 'overlay'
color = '9effffff'
visibility = '1'

prop = ds[prop_name].cf.isel(xi_rho=slice(200, None))

# get lat/lon
lon = prop[ds.salt.cf.coordinates['longitude'][0]].values
lat = prop[ds.salt.cf.coordinates['latitude'][0]].values

# get time coordinate boundaries
dt = ds.ocean_time[1] - ds.ocean_time[0]
time_starts = ds.ocean_time.values
time_stops = np.hstack((ds.ocean_time.values[1:], ds.ocean_time[-1].values+dt))

# set aspect ratio for figure, so points are roughly even
geo_aspect = np.cos(lat.mean()*np.pi/180.0)
xsize = lon.ptp()*geo_aspect
ysize = lat.ptp()

aspect = ysize/xsize
if aspect > 1.0:
    figsize = (10.0/aspect, 10.0)
else:
    figsize = (10.0, 10.0*aspect)

fig = plt.figure(figsize=figsize, dpi=pixels//10, facecolor=None, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])

kmz_name = f'txla_sss_{timestamp}.kmz'
f = zipfile.ZipFile(kmz_name, 'w', compression=zipfile.ZIP_DEFLATED)

kml_text = kml_preamble

for frame in range(prop.shape[0]):
    tstart =str(time_starts[frame])
    tstop = str(time_stops[frame])
    print(f'Writing frame {frame} {tstart}, {tstop}')
    ax.cla()
    pc = ax.pcolor(lon, lat, prop[frame,-1].values, 
                   vmin=vmin, vmax=vmax, cmap=cmap,
                   shading='nearest')
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_axis_off()
    icon = f'overlay_{frame}.png'
    plt.savefig(icon)
    kml_text += kml_frame.replace('__NAME__', name)\
                         .replace('__COLOR__', color)\
                         .replace('__VISIBILITY__', visibility)\
                         .replace('__SOUTH__', str(lat.min()))\
                         .replace('__NORTH__', str(lat.max()))\
                         .replace('__EAST__', str(lon.max()))\
                         .replace('__WEST__', str(lon.min()))\
                         .replace('__FRAME__', icon)\
                         .replace('__TIMEBEGIN__', tstart)\
                         .replace('__TIMEEND__', tstop)

    f.write(icon)
    os.remove(icon)

# legend
fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)
cax = fig.add_axes([0.0, 0.05, 0.2, 0.90])
cb = plt.colorbar(pc, cax=cax)
cb.set_label(units, color='0.9')
for lab in cb.ax.get_yticklabels():
    plt.setp(lab, 'color', '0.9')

plt.savefig('legend.png')
f.write('legend.png')
os.remove('legend.png')

kml_text += kml_legend

kml_text += kml_closing

f.writestr('overlay.kml', kml_text)

f.close()



N=200
u = xarray.DataArray(data=ds.u[:, -1, :-1, N:].values,
                coords=dict(lon=(["y", "x"], ds.lon_psi.values[:, N:]),
                            lat=(["y", "x"], ds.lat_psi.values[:, N:]),
                            time=ds.ocean_time),
                dims=['ocean_time', 'y', 'x'])
v = xarray.DataArray(data=ds.v[:, -1, :, N:-1].values,
                coords=dict(lon=(["y", "x"], ds.lon_psi.values[:, N:]),
                            lat=(["y", "x"], ds.lat_psi.values[:, N:]),
                            time=ds.ocean_time),
                dims=['ocean_time', 'y', 'x'])

ds_parcels = xarray.Dataset({'u': u, 'v': v})

variables = {'U': 'u', 'V': 'v'}
dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
fieldset = FieldSet.from_xarray_dataset(ds_parcels, variables, dimensions, 
                                        interp_method='cgrid_velocity',
                                        gridindexingtype='nemo')

nparticles = 30
lon, lat = np.meshgrid(np.linspace(-94, -91.5, nparticles), np.linspace(28., 29.3, nparticles))
pset = ParticleSet(fieldset=fieldset, 
                   pclass=JITParticle,
                   lon=lon, lat=lat)   

output_file = pset.ParticleFile(name='tracks.nc', outputdt=timedelta(hours=1))
pset.write=True
pset.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
             endtime=ds_parcels.time[-1].values,    # the total length of the run
             dt=timedelta(minutes=5),      # the timestep of the kernel
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

output_file.export()

floats=xarray.open_dataset('tracks.nc')

marker_style = '''  <Style id="mystyle">
    <IconStyle>
      <scale>0.5</scale>
      <Icon>
        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
      </Icon>
    </IconStyle>
  </Style>
'''

with open(f'tracks.kml', 'w') as fp:
    fp.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fp.write('<kml xmlns="http://www.opengis.net/kml/2.2"\n')
    fp.write('     xmlns:gx="http://www.google.com/kml/ext/2.2">\n')
    fp.write('<Folder>\n')
    fp.write(' <name>Numerical Drifter Tracks</name>\n')
    for track_id, track in floats.groupby('traj'):
        fp.write(' <Placemark>\n')
        fp.write(f'  <name></name>\n')
        fp.write(marker_style)
        fp.write('  <gx:Track>\n')
        for time in track.time:
            fp.write(f'    <when>{time.values}</when>\n')
        for lon, lat in zip(track.lon.values, track.lat.values):
            fp.write(f'    <gx:coord>{lon},{lat}</gx:coord>\n')
        fp.write('  </gx:Track>\n')
        fp.write(' </Placemark>\n')

    fp.write('</Folder>\n')
    fp.write('</kml>\n')

kmz_name = f'txla_tracks_{timestamp}.kmz'
with zipfile.ZipFile(kmz_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(f'tracks.kml')

