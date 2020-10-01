import sys
import os
import numpy as np
import scipy as sp
import scipy.interpolate as interp
import math
from types import SimpleNamespace

from datetime import datetime

try:
  import netCDF4 as netCDF
except:
  import netCDF3 as netCDF

croco_input = "croco_rst.nc"
croco_output = "croco_ini.nc"
  
nc_in = netCDF.Dataset(croco_input, 'r')
nc_out = netCDF.Dataset(croco_output, 'a')

salt = nc_in.variables['salt'][:]
temp = nc_in.variables['temp'][:]
u = nc_in.variables['u'][:]
v = nc_in.variables['v'][:]
ubar = nc_in.variables['ubar'][:]
vbar = nc_in.variables['vbar'][:]
zeta = nc_in.variables['zeta'][:]

nc_out.variables['salt'][:] = salt
nc_out.variables['temp'][:] = temp
nc_out.variables['u'][:] = u
nc_out.variables['v'][:] = v
nc_out.variables['ubar'][:] = ubar
nc_out.variables['vbar'][:] = vbar
nc_out.variables['zeta'][:] = zeta