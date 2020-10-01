# encoding: utf-8
# A python version of make_grid for ROMS/CROCO
# A direct port from CROCOTOOLS v1.1 by Dylan F. Bailey 2020
# write_nc_var from PyTools

# TODO
# make_nested_grid

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

np.seterr(divide='ignore', invalid='ignore')

def make_CROCO_grid(grd,topofile,title,filename='croco_grd.nc'):

    def write_nc_var(var, name, dimensions, long_name=None, units=None):
        nc.createVariable(name, 'f8', dimensions)
        if long_name is not None:
            nc.variables[name].long_name = long_name
        if units is not None:
            nc.variables[name].units = units
        nc.variables[name][:] = var
        print(' ... wrote ', name)
        
    def rho2uvp(rfield):
        Mp, Lp = np.shape(rfield)
        M = Mp-1
        L = Lp-1     
        vfield = 0.5*(rfield[0:M,:]+rfield[1:Mp,:])
        ufield = 0.5*(rfield[:,0:L]+rfield[:,1:Lp])
        pfield = 0.5*(ufield[0:M,:]+ufield[1:Mp,:])
        return [ufield, vfield, pfield]
    
    def uvp_mask(rfield):
        Mp, Lp = np.shape(rfield)
        M = Mp-1
        L = Lp-1     
        vfield = rfield[0:M,:]*rfield[1:Mp,:]
        ufield = rfield[:,0:L]*rfield[:,1:Lp]
        pfield = ufield[0:M,:]*ufield[1:Mp,:]
        return [ufield, vfield, pfield]
        
    def spheric_dist(lat1,lat2,lon1,lon2):
        # Earth radius
        R = 6367442.76
        
        #  Determine proper longitudinal shift.
        l = np.abs(lon2-lon1)
        l[l>=180] = 360-l[l>=180]

        #  Convert Decimal degrees to radians.
        deg2rad = math.pi/180
        lat1 = lat1*deg2rad
        lat2 = lat2*deg2rad
        l = l*deg2rad
        
        #  Compute the distances
        return R*np.arcsin(np.sqrt(((np.sin(l)*np.cos(lat2))**2)+(((np.sin(lat2)*np.cos(lat1))-(np.sin(lat1)*np.cos(lat2)*np.cos(l)))**2)))
    
    def metrics(latu, lonu, latv, lonv):
        Mp, L = np.shape(latu)
        M, Lp = np.shape(latv)
        Lm = L-1
        Mm = M-1
        
        # pm and pn
        dx = np.zeros((Mp,Lp))
        dy = np.zeros((Mp,Lp))
        
        dx[:,1:L] = spheric_dist(latu[:,0:Lm],latu[:,1:L],lonu[:,0:Lm],lonu[:,1:L])
        dx[:,0] = dx[:,1]
        dx[:,Lp-1] = dx[:,L-1]

        dy[1:M,:] = spheric_dist(latv[0:Mm,:],latv[1:M,:],lonv[0:Mm,:],lonv[1:M,:])
        dy[0,:] = dy[1,:]
        dy[Mp-1,:] = dy[M-1,:]

        pm = 1/dx
        pn = 1/dy

        # dndx and dmde
        
        dndx = np.zeros((Mp,Lp))
        dmde = np.zeros((Mp,Lp))
        
        dndx[1:M,1:L] = 0.5*(1/pn[1:M,2:Lp] - 1/pn[1:M,0:Lm])
        dmde[1:M,1:L] = 0.5*(1/pm[2:Mp,1:L] - 1/pm[0:Mm,1:L])
        dndx[0,:] = 0
        dndx[Mp-1,:] = 0
        dndx[:,0] = 0
        dndx[:,Lp-1] = 0
        dmde[0,:] = 0
        dmde[Mp-1,:] = 0
        dmde[:,0] = 0
        dmde[:,Lp-1] = 0
        
        return [pm, pn, dndx, dmde]
    
    def get_angle(latu,lonu):
        
        #Only WGS84 implememnted. Modify as needed.
        
        A = 6378137.
        E = 0.081819191
        B = math.sqrt(A**2 - (A*E)**2)
        EPS = E*E/(1-E*E)
        
        latu = latu*math.pi/180     # convert to radians
        lonu = lonu*math.pi/180

        latu[latu==0] = EPS  # Fixes some nasty 0/0 cases in the
                             # geodesics stuff
        M, L = np.shape(latu)

        PHI1 = latu[0:M,0:L-1]    # endpoints of each segment
        XLAM1 = lonu[0:M,0:L-1]
        PHI2 = latu[0:M,1:L]
        XLAM2 = lonu[0:M,1:L]

        # wiggle lines of constant lat to prevent numerical probs.
        PHI2[PHI1==PHI2] = PHI2[PHI1==PHI2] + 1e-14
        
        # wiggle lines of constant lon to prevent numerical probs.
        XLAM2[XLAM1==XLAM2] = XLAM2[XLAM1==XLAM2] + 1e-14

        # COMPUTE THE RADIUS OF CURVATURE IN THE PRIME VERTICAL FOR
        # EACH POINT
        xnu1 = A/np.sqrt(1.0-(E*np.sin(PHI1))**2)
        xnu2 = A/np.sqrt(1.0-(E*np.sin(PHI2))**2)

        # COMPUTE THE AZIMUTHS.  azim  IS THE AZIMUTH AT POINT 1
        # OF THE NORMAL SECTION CONTAINING THE POINT 2
        TPSI2 = (1-E*E)*np.tan(PHI2) + E*E*xnu1*np.sin(PHI1)/(xnu2*np.cos(PHI2))

        # SOME FORM OF ANGLE DIFFERENCE COMPUTED HERE??
        DLAM = XLAM2-XLAM1
        CTA12 = (np.cos(PHI1)*TPSI2 - np.sin(PHI1)*np.cos(DLAM))/np.sin(DLAM)
        azim = np.arctan(1/CTA12)

        #  GET THE QUADRANT RIGHT
        DLAM2 = (abs(DLAM)<math.pi)*DLAM + (DLAM>=math.pi)*(-2*math.pi+DLAM) + (DLAM<=-math.pi)*(2*math.pi+DLAM)
        azim = azim+(azim<-math.pi)*2*math.pi-(azim>=math.pi)*2*math.pi
        azim = azim+math.pi*np.sign(-azim)*(np.sign(azim) != np.sign(DLAM2))
        
        angle = np.zeros((M,L+1))
        angle[:,1:L] = (math.pi/2)-azim[:,0:L-1]
        angle[:,0] = angle[:,1]
        angle[:,L] = angle[:,L-1] 

        return angle
    
    def add_topo(lon,lat,pm,pn,topofile):
  
        # Get ROMS averaged resolution
        dx = np.mean(np.mean(1/pm))
        dy = np.mean(np.mean(1/pn))
        
        dx_roms = np.mean([dx,dy]) # This is why numpy sucks at matrices compared to Matlab
        #disp(['   ROMS resolution : ',num2str(dx_roms/1000,3),' km'])
        
        dl = np.max([1,2*(dx_roms/(60*1852))])
        lonmin = np.min(np.min(lon))-dl
        lonmax = np.max(np.max(lon))+dl
        latmin = np.min(np.min(lat))-dl
        latmax = np.max(np.max(lat))+dl
        
        #  open the topo file
        nc = netCDF.Dataset(topofile, 'r')
        tlon = nc.variables['lon'][:]
        tlat = nc.variables['lat'][:]
        
        #  get a subgrid
        j = np.where((tlat>=latmin) & (tlat<=latmax))[0]
        i1 = np.where((tlon-360>=lonmin) & (tlon-360<=lonmax))[0]
        i2 = np.where((tlon>=lonmin) & (tlon<=lonmax))[0]
        i3 = np.where((tlon+360>=lonmin) & (tlon+360<=lonmax))[0]
        x = np.concatenate((tlon[i1]-360,tlon[i2],tlon[i3]+360))
        y = tlat[j]
        
        #  Read data
        if i2.size != 0:
            try:
                topo = -nc.variables['elevation'][j,i2];
            except:
                topo = -nc.variables['topo'][j,i2];
        else:
            topo = []

        if i1.size != 0:
            topo = np.concatenate((-nc.variables['elevation'][j,i1],topo),axis=1);
    
        if i3.size != 0:
            topo = np.concatenate((topo,-nc.variables['elevation'][j,i3]),axis=1);
        
        # Get TOPO averaged resolution
        R = 6367442.76
        deg2rad = math.pi/180
        dg = np.mean(x[1:]-x[0:-1])
        dphi = y[1:]-y[0:-1]
        dy = R*deg2rad*dphi
        dx = R*deg2rad*dg*np.cos(deg2rad*y)
        dx_topo = np.mean(np.concatenate((dx, dy)))
        #disp(['   Topography data resolution : ',num2str(dx_topo/1000,3),' km'])
        
        # Degrade TOPO resolution
        n = 0
        while dx_roms>(dx_topo):
            
            n = n+1
            x = 0.5*(x[1:]+x[0:-1])
            x = x[0::2]
            y = 0.5*(y[1:]+y[0:-1]);
            y = y[0::2]
            
            topo = 0.25*(topo[1:,0:-1] + topo[1:,1:] + topo[0:-1,0:-1] + topo[0:-1,1:])
            topo = topo[0::2,0::2]
            
            dg = np.mean(x[1:]-x[0:-1])
            dphi = y[1:]-y[0:-1]
            dy = R*deg2rad*dphi
            dx = R*deg2rad*dg*np.cos(deg2rad*y)
            
            dx_topo = np.mean(np.concatenate((dx, dy)))
        
        #disp(['   Topography resolution halved ',num2str(n),' times'])
        #disp(['   New topography resolution : ',num2str(dx_topo/1000,3),' km'])
       
        #  interpolate the topo
        xx, yy = np.meshgrid(x,y)
        
        # From the excellent post at 
        # https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
        h = interp.griddata(np.array([xx.ravel(),yy.ravel()]).T,topo.ravel(),(lon,lat),method='cubic')

        return h
            
    def process_mask(maskin):
        
        maskout = maskin
        
        M,L = np.shape(maskout);
        Mm = M-1
        Lm = L-1
        Mmm = Mm-1
        Lmm = Lm-1
        
        neibmask = 0*maskout
        neibmask[1:Mm,1:Lm] = maskout[0:Mmm,1:Lm]+maskout[2:M,1:Lm]+maskout[1:Mm,0:Lmm]+maskout[1:Mm,2:L]
        
        while np.sum(np.sum(((neibmask[1:Mm,1:Lm]>=3) & (maskout[1:Mm,1:Lm]==0))|((neibmask[1:Mm,1:Lm]<=1) & (maskout[1:Mm,1:Lm]==1)))) > 0:
        
            maskout[(neibmask>=3) & (maskout==0)] = 1;
            maskout[(neibmask<=1) & (maskout==1)] = 0;

            maskout[0,1:Lm] = maskout[1,1:Lm];
            maskout[M-1,1:Lm] = maskout[Mm-1,1:Lm];
            maskout[1:Mm,0] = maskout[1:Mm,1];
            maskout[1:Mm,L-1] = maskout[1:Mm,Lm-1];

            maskout[0,0] = min([maskout[0,1],maskout[1,0]]);
            maskout[M-1,0] = min([maskout[M-1,1],maskout[Mm-1,0]]);
            maskout[0,L-1] = min([maskout[0,Lm-1],maskout[1,L-1]]);
            maskout[M-1,L-1] = min([maskout[M-1,Lm-1],maskout[Mm-1,L-1]]);

            neibmask[1:Mm,1:Lm] = maskout[0:Mmm,1:Lm]+maskout[2:M,1:Lm]+maskout[1:Mm,0:Lmm]+maskout[1:Mm,2:L];

        # Be sure that there is no problem close to the boundaries
        
        maskout[:,0] = maskout[:,1];
        maskout[:,L-1] = maskout[:,Lm-1];
        maskout[0,:] = maskout[1,:];
        maskout[M-1,:] = maskout[Mm-1,:];
        
        return maskout
    
    def smoothgrid_new(h,maskr,masku,maskv,maskp,hmin,hmax_coast,hmax,r_max,n_filter_deep_topo,n_filter_final):
    
        def log_topo_filter(h,maskr,masku,maskv,maskr_ext,hmin,hmax_coast,r_max):

            def np_div(x,np_y): # x is integer, y is numpy array
                xi, yi = np.shape(np_y)
                np_x = x*np.ones((xi,yi))
                return np.divide(np_x,np_y)
            
            def np_mul(x,np_y): # x is integer, y is numpy array
                xi, yi = np.shape(np_y)
                np_x = x*np.ones((xi,yi))
                return np.multiply(np_x,np_y)
            
            # Apply a selective filter on log(h) to reduce grad(h)/h.
            # Adapted from Alexander Shchepetkin fortran smooth.F program
             
            # Addition: constraint on maximum depth for the closest point to the mask
            # This prevent isobaths to run below the mask, resulting in current detachment. 
             
            OneEights = 1/8
            OneThirtyTwo = 1/32

            r = rfact(h,masku,maskv)
            cff = 1.4
            r_max = r_max/cff
            i = 0

            while r>(r_max*cff):

                i = i+1

                Lgh = np.log(h/hmin)
                Lgh[hmin==0] = 0

                lgr_max = np.log((1+r_max)/(1-r_max))
                lgr1_max = lgr_max*math.sqrt(2.) 

                grad = (Lgh[:,1:]-Lgh[:,0:-1])
                cr = np.absolute(grad)
                FX = np_mul(grad,(1-np_div(lgr_max,cr)))
                FX[cr<=lgr_max] = 0.

                grad = (Lgh[1:,1:]-Lgh[0:-1,0:-1])
                cr = np.absolute(grad)
                FX1 = np_mul(grad,(1-np_div(lgr1_max,cr)))
                FX1[cr<=lgr1_max] = 0.

                grad = (Lgh[1:,:]-Lgh[0:-1,:])
                cr = np.absolute(grad)
                FE = np_mul(grad,(1-np_div(lgr_max,cr)))
                FE[cr<=lgr_max] = 0.

                grad = (Lgh[1:,0:-1]-Lgh[0:-1,1:])
                cr = np.absolute(grad)
                FE1 = np_mul(grad,(1-np_div(lgr1_max,cr)))
                FE1[cr<=lgr1_max] = 0.

                Lgh[1:-1,1:-1] = Lgh[1:-1,1:-1]+OneEights*(FX[1:-1,1:]-FX[1:-1,0:-1]+FE[1:,1:-1]-FE[0:-1,1:-1])+OneThirtyTwo*(FX1[1:,1:]-FX1[0:-1,0:-1]+FE1[1:,0:-1]-FE1[0:-1,1:])

                Lgh[0,:] = Lgh[1,:]
                Lgh[-1,:] = Lgh[-2,:]
                Lgh[:,0] = Lgh[:,1]
                Lgh[:,-1] = Lgh[:,-2]

                h = hmin*np.exp(Lgh)

                h[(maskr_ext<0.5) & (h>hmax_coast)] = hmax_coast

                r = rfact(h,masku,maskv)

            return h

        def rfact(h,masku,maskv):
            rx = np.absolute(h[:,1:]-h[:,0:-1])/(h[:,1:]+h[:,0:-1])
            ry = np.absolute(h[1:,:]-h[0:-1,:])/(h[1:,:]+h[0:-1,:])
            rx_max = np.max(rx[masku==1])
            ry_max = np.max(ry[maskv==1])
            r = np.max([rx_max,ry_max])
            return r

        def hanning_smoother(h):
            M, L = np.shape(h);
            Mm = M-1
            Mmm = M-2
            Lm = L-1
            Lmm = L-2
            h[1:Mm,1:Lm] = 0.125*(h[0:Mmm,1:Lm]+h[2:M,1:Lm]+h[1:Mm,0:Lmm]+h[1:Mm,2:L]+4*h[1:Mm,1:Lm])
            h[0,:] = h[1,:];
            h[M-1,:] = h[Mm-1,:]
            h[:,0] = h[:,1]
            h[:,L-1] = h[:,Lm-1]
            return h
        
        def hanning_smoother_coef2d(h,coef):
            M, L = np.shape(h)
            Mm = M-1
            Mmm = M-2
            Lm = L-1
            Lmm = L-2
            h[1:Mm,1:Lm] = coef[1:Mm,1:Lm]*(h[0:Mmm,1:Lm]+h[2:M,1:Lm]+h[1:Mm,0:Lmm]+h[1:Mm,2:L])+(1-4*coef[1:Mm,1:Lm])*h[1:Mm,1:Lm]
            h[0,:] = h[1,:]
            h[M-1,:] = h[Mm-1,:]
            h[:,0] = h[:,1]
            h[:,L-1] = h[:,Lm-1]
            return h
        
        def hann_window(h):
            OneFours = 1/4 
            OneEights = 1/8
            OneSixteens = 1/16
            h[1:-1,1:-1] = OneFours*h[1:-1,1:-1]+OneEights*(h[0:-2,1:-1]+h[2:,1:-1]+h[1:-1,0:-2]+h[1:-1,2:])+OneSixteens*(h[0:-2,0:-2]+h[2:,2:]+h[0:-2,2:]+h[2:,0:-2])     
            h[0,:] = h[1,:]
            h[-1,:] = h[-2,:]
            h[:,0] = h[:,1]
            h[:,-1] = h[:,-2]
            return h
        
        maskr_ext = hann_window(maskr.copy())
        maskr_ext[maskr_ext<1] = 0
        
        # Cut topography and flood dry cells momentarily
        
        h[h<hmin] = hmin
        h[h>hmax] = hmax
        
        # 1: Deep Ocean Filter
        
        if n_filter_deep_topo>=1:
            #disp(' Apply a filter on the Deep Ocean to reduce isolated seamounts :')
            #disp(['   ',num2str(n_filter_deep_topo),' pass of a selective filter.'])
        
            #  Build a smoothing coefficient that is a linear function 
            #  of a smooth topography.
        
            coef = h
            for i in range(1,8):
                coef = hann_window(coef.copy())                  # coef is a smoothed bathy
 
            coef = 0.125*(coef/np.max(np.max(coef)));         # rescale the smoothed bathy
          
        for i in range(1,n_filter_deep_topo):
            h = hanning_smoother_coef2d(h.copy(),coef.copy())       # smooth with available coef
            h[(maskr_ext<0.5) & (h>hmax_coast)] = hmax_coast
        
        #  Apply a selective filter on log(h) to reduce grad(h)/h.
        #disp(' Apply a selective filter on log(h) to reduce grad(h)/h :')
        h = log_topo_filter(h.copy(),maskr,masku,maskv,maskr_ext,hmin,hmax_coast,r_max)
        
        #  Smooth the topography again to prevent 2D noise
        
        if n_filter_final>1:
            #disp(' Smooth the topography a last time to prevent 2DX noise:')
            #disp(['   ',num2str(n_filter_final),' pass of a hanning smoother.'])
            for i in range(1,n_filter_final):
                h = hann_window(h.copy())
                h[(maskr_ext<0.5) & (h>hmax_coast)] = hmax_coast

        h[h<hmin] = hmin
        
        return h
    
    
    lonmin = grd[0]
    lonmax = grd[1]
    latmin = grd[2]
    latmax = grd[3]
    dl = grd[4]
    hmin = grd[5]
    hmax_coast = grd[6]
    hmax = grd[7]
    rtarget = grd[8]
    n_filter_deep_topo = grd[9]
    n_filter_final = grd[10]
    
    lonr = np.arange(lonmin, lonmax+dl, dl)
    
    i = 0
    latr = np.empty(1)
    latr[i] = latmin
    while latr[i] <= latmax:
        i = i+1;
        latr = np.append(latr,latr[i-1]+dl*math.cos(latr[i-1]*math.pi/180.))
    
    Lonr, Latr = np.meshgrid(lonr, latr)
    
    Lonu, Lonv, Lonp = rho2uvp(Lonr.copy())
    Latu, Latv, Latp = rho2uvp(Latr.copy())
    
    M, L = np.shape(Latp)
    
    pm, pn, dndx, dmde = metrics(Latu.copy(), Lonu.copy(), Latv.copy(), Lonv.copy())
    
    xr = 0*pm.copy()
    yr = xr.copy()
    for i in range(0,L):
      xr[:,i+1]=xr[:,i]+2/(pm[:,i+1]+pm[:,i])
    
    for j in range(0,M):
      yr[j+1,:]=yr[j,:]+2/(pn[j+1,:]+pn[j,:])
    
    xu, xv, xp = rho2uvp(xr.copy())
    yu, yv, yp = rho2uvp(yr.copy())
    dx = 1/pm
    dy = 1/pn
    dxmax = np.max(np.max(dx/1000.))
    dxmin = np.min(np.min(dx/1000.))
    dymax = np.max(np.max(dy/1000.))
    dymin = np.min(np.min(dy/1000.))
    
    #  Angle between XI-axis and the direction
    #  to the EAST at RHO-points [radians].
    angle = get_angle(Latu.copy(),Lonu.copy())
    
    #  Coriolis parameter
    f = 4*math.pi*np.sin(math.pi*Latr/180.)*366.25/(24*3600*365.25)
    
    # Add topography
    hin = add_topo(Lonr.copy(),Latr.copy(),pm.copy(),pn.copy(),topofile)
    
    # Compute the mask
    maskin = hin.copy()
    maskin[maskin>0] = 1
    maskin[maskin<=0] = 0
    maskr = process_mask(maskin.copy())
    masku, maskv, maskp = uvp_mask(maskr.copy())
    
    h = smoothgrid_new(hin.copy(),maskr.copy(),masku.copy(),maskv.copy(),maskp.copy(),hmin,hmax_coast,hmax,rtarget,n_filter_deep_topo,n_filter_final) 
    hraw = h.copy()
    
    xl = 0
    el = 0
    depthmin = 0
    depthmax = 0
    spherical = 'T'
    
    nc = netCDF.Dataset(filename, 'w', format='NETCDF3_64BIT')
    nc.title = title
    nc.author = 'make_grid.py'
    nc.date = datetime.now().isoformat()
    nc.type = 'CROCO grid file'

    nc.createDimension('one', 1)
    nc.createDimension('two', 2)
    nc.createDimension('four', 4)
    nc.createDimension('bath', 1)
    
    nc.createDimension('xi_rho', L+1)
    nc.createDimension('xi_u', L)
    nc.createDimension('xi_v', L+1)
    nc.createDimension('xi_psi', L)

    nc.createDimension('eta_rho', M+1)
    nc.createDimension('eta_u', M+1)
    nc.createDimension('eta_v', M)
    nc.createDimension('eta_psi', M)
    
    write_nc_var(h, 'h', ('eta_rho', 'xi_rho'), 'Bathymetry at RHO-points', 'meter')
    write_nc_var(hraw, 'hraw', ('bath', 'eta_rho', 'xi_rho'), 'Working bathymetry at RHO-points', 'meter')
    write_nc_var(f, 'f', ('eta_rho', 'xi_rho'), 'Coriolis parameter at RHO-points', 'second-1')
    write_nc_var(pm, 'pm', ('eta_rho', 'xi_rho'), 'Curvilinear coordinate metric in XI', 'meter-1')
    write_nc_var(pn, 'pn', ('eta_rho', 'xi_rho'), 'Curvilinear coordinate metric in ETA', 'meter-1')
    write_nc_var(dndx, 'dndx', ('eta_rho', 'xi_rho'), 'XI derivative of inverse metric factor pn', 'meter')
    write_nc_var(dmde, 'dmde', ('eta_rho', 'xi_rho'), 'ETA derivative of inverse metric factor pm', 'meter')
    write_nc_var(xl, 'xl', ('one'), 'domain length in the XI-direction', 'meter')
    write_nc_var(el, 'el', ('one'), 'domain length in the ETA-direction', 'meter')
    write_nc_var(depthmin, 'depthmin', ('one'), 'Shallow bathymetry clipping depth', 'meter')
    write_nc_var(depthmax, 'depthmax', ('one'), 'Deep bathymetry clipping depth', 'meter')

    write_nc_var(xr, 'x_rho', ('eta_rho', 'xi_rho'), 'x location of RHO-points', 'meter')
    write_nc_var(yr, 'y_rho', ('eta_rho', 'xi_rho'), 'y location of RHO-points', 'meter')
    write_nc_var(xu, 'x_u', ('eta_u', 'xi_u'), 'x location of U-points', 'meter')
    write_nc_var(yu, 'y_u', ('eta_u', 'xi_u'), 'y location of U-points', 'meter')
    write_nc_var(xv, 'x_v', ('eta_v', 'xi_v'), 'x location of V-points', 'meter')
    write_nc_var(yv, 'y_v', ('eta_v', 'xi_v'), 'y location of V-points', 'meter')
    write_nc_var(xp, 'x_psi', ('eta_psi', 'xi_psi'), 'x location of PSI-points', 'meter')
    write_nc_var(yp, 'y_psi', ('eta_psi', 'xi_psi'), 'y location of PSI-points', 'meter')

    write_nc_var(Lonr, 'lon_rho', ('eta_rho', 'xi_rho'), 'Longitude of RHO-points', 'degree_east')
    write_nc_var(Latr, 'lat_rho', ('eta_rho', 'xi_rho'), 'Latitude of RHO-points', 'degree_north')
    write_nc_var(Lonu, 'lon_u', ('eta_u', 'xi_u'), 'Longitude of U-points', 'degree_east')
    write_nc_var(Latu, 'lat_u', ('eta_u', 'xi_u'), 'Latitude of U-points', 'degree_north')
    write_nc_var(Lonv, 'lon_v', ('eta_v', 'xi_v'), 'Longitude of V-points', 'degree_east')
    write_nc_var(Latv, 'lat_v', ('eta_v', 'xi_v'), 'Latitude of V-points', 'degree_north')
    write_nc_var(Lonp, 'lon_psi', ('eta_psi', 'xi_psi'), 'Longitude of PSI-points', 'degree_east')
    write_nc_var(Latp, 'lat_psi', ('eta_psi', 'xi_psi'), 'Latitude of PSI-points', 'degree_north')  
    
    write_nc_var(angle, 'angle', ('eta_rho', 'xi_rho'), 'Angle between XI-axis and EAST', 'radians')
    
    nc.createVariable('spherical', 'c', 'one')
    nc.variables['spherical'].long_name = 'Grid type logical switch'
    nc.variables['spherical'].option_T = 'spherical'
    nc.variables['spherical'][:] = spherical

    nc.createVariable('mask_rho', 'f8', ('eta_rho', 'xi_rho'))
    nc.variables['mask_rho'].long_name = 'Mask on RHO-points'
    nc.variables['mask_rho'].option_0 = 'land'
    nc.variables['mask_rho'].option_1 = 'water'
    nc.variables['mask_rho'][:] = maskr
    
    nc.createVariable('mask_u', 'f8', ('eta_u', 'xi_u'))
    nc.variables['mask_u'].long_name = 'Mask on U-points'
    nc.variables['mask_u'].option_0 = 'land'
    nc.variables['mask_u'].option_1 = 'water'
    nc.variables['mask_u'][:] = masku
    
    nc.createVariable('mask_v', 'f8', ('eta_v', 'xi_v'))
    nc.variables['mask_v'].long_name = 'Mask on V-points'
    nc.variables['mask_v'].option_0 = 'land'
    nc.variables['mask_v'].option_1 = 'water'
    nc.variables['mask_v'][:] = maskv
    
    nc.createVariable('mask_psi', 'f8', ('eta_psi', 'xi_psi'))
    nc.variables['mask_psi'].long_name = 'Mask on PSI-points'
    nc.variables['mask_psi'].option_0 = 'land'
    nc.variables['mask_psi'].option_1 = 'water'
    nc.variables['mask_psi'][:] = maskp

    nc.close()
    
    print('Done!')
    print('LLm:', M-1,' MMm:', L-1)
    
    return
    
#############################################################################

if __name__ == '__main__':

    topofile = 'GEBCO.nc'
    title = 'Agulhas Bank'
    grid_file = '/server/DISK3/SAnLX/INPUT/croco_grd.nc'
    lonmin =  19
    lonmax =  29
    latmin = -37 #-36
    latmax = -32
    dl = 1/36
    hmin = 10
    
    if False:
        grid_file = '/server/DISK3/SAnL2/INPUT/croco_grd.nc'
        lonmin =  20.
        lonmax =  29.
        latmin = -36.031798
        latmax = -32.009223
        dl = 1/36
        hmin = 30
    
    hmax_coast = 5000
    hmax = 5000
    rtarget = 0.25
    n_filter_deep_topo = 4
    n_filter_final = 4

    grd = [lonmin,lonmax,latmin,latmax,dl,hmin,hmax_coast,hmax,rtarget,n_filter_deep_topo,n_filter_final]
    make_CROCO_grid(grd,topofile,title,grid_file)