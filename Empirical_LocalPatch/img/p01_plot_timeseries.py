#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 01:49:20 2023

@author: abdul.moiz
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import re

#Custom Fuctions
sys.path.append('../')
import params as pm
import read_grdc

def test_plot(np_array,mask=-9999.0):
    masked_array=np.ma.masked_where(np_array==-9999.0,np_array)
    fig,ax=plt.subplots()
    ax.imshow(masked_array)

def read_params(params):
    params=pd.read_csv(params,delim_whitespace=True,header=None)
    nx=int(params[0][0])
    ny=int(params[0][1])
    gsize=float(params[0][3])
    west=float(params[0][4])
    east=float(params[0][5])
    south=float(params[0][6])
    north=float(params[0][7])
    return nx,ny,gsize,west,east,south,north

def read_regional_maps(map_name_path,nx,ny):
    nextxy = np.fromfile(os.path.join(map_name_path,'nextxy.bin'),np.int32).reshape(2,ny,nx)
    nextx = nextxy[0]
    nexty = nextxy[1]
    rivwth = np.fromfile(os.path.join(map_name_path,'rivwth.bin'),np.float32).reshape(ny,nx)
    rivhgt = np.fromfile(os.path.join(map_name_path,'rivhgt.bin'),np.float32).reshape(ny,nx)
    rivlen = np.fromfile(os.path.join(map_name_path,'rivlen.bin'),np.float32).reshape(ny,nx)
    elevtn = np.fromfile(os.path.join(map_name_path,'elevtn.bin'),np.float32).reshape(ny,nx)
    uparea = np.fromfile(os.path.join(map_name_path,'uparea.bin'),np.float32).reshape(ny,nx)
    lonlat = np.fromfile(os.path.join(map_name_path,'lonlat.bin'),np.float32).reshape(2,ny,nx)
    return nextx,nexty,rivwth,rivhgt,rivlen,elevtn,uparea,lonlat

def latlon_river(rivername,ix,iy,mapname="glb_06min",nYY=1800,nXX=3600):
    #global lllat, urlat, lllon, urlon
    lonlat = pm.CaMa_dir()+"/map/glb_"+mapname[-5::]+"/lonlat.bin"
    lonlat = np.fromfile(lonlat,np.float32).reshape(2,nYY,nXX)
    llon=lonlat[0,iy-1,ix-1]
    llat=lonlat[1,iy-1,ix-1]
    adj=20.0
    lllat, urlat, lllon, urlon = max(llat-adj,-90.),min(llat+adj,90.),max(llon-adj,-180.),min(llon+adj,180.)
    if rivername=="LENA":
        lllat = 50.
        urlat = 80.
        lllon = 100.
        urlon = 145.
    if rivername=="NIGER":
        lllat = 0.
        urlat = 25.
        lllon = -10.
        urlon = 15.
    if rivername=="AMAZONAS" or rivername=="AMAZON":
        lllat = -20.
        urlat = 10.
        lllon = -80.
        urlon = -45.
    if rivername=="MADEIRA":
        lllat = -20.
        urlat = 10.
        lllon = -80.
        urlon = -45.
    if rivername=="MEKONG":
        lllat = 10.
        urlat = 35.
        lllon = 90.
        urlon = 120.
    if rivername=="MISSISSIPPI":
        lllat = 30.
        urlat = 50.
        lllon = -115.
        urlon = -75.
    if rivername=="OB":
        lllat = 40.
        urlat = 70.
        lllon = 55.
        urlon = 95.
    if rivername=="CONGO":
        lllat = -15.
        urlat = 10.
        lllon = 10.
        urlon = 35.
    if rivername=="INDUS":
        lllat = 20.
        urlat = 40.
        lllon = 60.
        urlon = 80.
    if rivername=="VOLGA":
        lllat = 40.
        urlat = 65.
        lllon = 30.
        urlon = 70.
    if rivername=="NILE":
        lllat = -5.
        urlat = 30.
        lllon = 20.
        urlon = 40.
    if rivername=="YUKON":
        lllat = 55.
        urlat = 75.
        lllon = -165.
        urlon = -130.
    if rivername=="YELLOW":
        lllat = 30.
        urlat = 45.
        lllon = 95.
        urlon = 120.
    if rivername=="MISSOURI":
        lllat = 20.
        urlat = 50.
        lllon = -115.
        urlon = -75.
    #if rivername not in ["LENA","NIGER","CONGO","OB","MISSISSIPPI","MEKONG","AMAZONAS","INDUS"]:
    #    adj=20.
    #    lllat, urlat, lllon, urlon = max(llat-adj,-90.),min(llat+adj,90.),max(llon-adj,-180.),min(llon+adj,180.)

    return lllat, urlat, lllon, urlon

def riveridname(rivername):
    river=rivername[0]+rivername[1::].lower()
    if rivername=="LENA":
        river="Lena"
    if rivername=="NIGER":
        river="Niger"
    if rivername=="AMAZONAS":
        river="Amazon"
    if rivername=="MADEIRA":
        river="Amazon"
    if rivername=="MEKONG":
        river="Mekong"  
    if rivername=="MISSISSIPPI":
        river="Mississippi"
    if rivername=="OB":
        river="Ob"
    if rivername=="CONGO":
        river="Congo"
    if rivername=="INDUS":
        river="Indus"
    if rivername=="ST._LAWRENCE":
        river="St Lawrence"
    if rivername=="BRAHMAPUTRA":
        river="Ganges-Brahamaputra"
    if rivername=="YELLOW":
        river="Huang He"
    if rivername=="MISSOURI":
        river="Mississippi"

    return river

def mk_dir(sdir):
  try:
    os.makedirs(sdir)
  except:
    pass


tag="%04d-%04d"%(pm.starttime()[0],pm.endtime()[0])
sfcelv = os.path.join(pm.out_dir(),'CaMa_out',pm.map_name()+'_'+pm.input_name(),'sfcelv'+tag+'.nc')
sfcelv = xr.open_dataset(sfcelv)

rmdtrnd = os.path.join(pm.out_dir(),'CaMa_out',pm.map_name()+'_'+pm.input_name(),'rmdtrnd'+tag+'.nc')
rmdtrnd = xr.open_dataset(rmdtrnd)

rmdsesn = os.path.join(pm.out_dir(),'CaMa_out',pm.map_name()+'_'+pm.input_name(),'rmdsesn'+tag+'.nc')
rmdsesn = xr.open_dataset(rmdsesn)

standardized = os.path.join(pm.out_dir(),'CaMa_out',pm.map_name()+'_'+pm.input_name(),'standardized'+tag+'.nc')
standardized = xr.open_dataset(standardized)


rivername = 'AMAZONAS'  # Change Here


# Read GRDC
pname=[]
xlist=[]
ylist=[]
river=[]
staid=[]

grdc_id,station_loc,x_list,y_list = read_grdc.get_grdc_loc_v396('AMAZON')
staid.append(grdc_id)
pname.append(station_loc)
xlist.append(x_list)
ylist.append(y_list)

river=([flatten for inner in river for flatten in inner])
pname=([flatten for inner in pname for flatten in inner])
xlist=([flatten for inner in xlist for flatten in inner])
ylist=([flatten for inner in ylist for flatten in inner])


pnum=len(pname)
for point in np.arange(pnum):
    print(point)
    fig, ax = plt.subplots(4,1,figsize=(6,6),sharex=True)
    sfcelv_pt = sfcelv.sfcelv[:,ylist[point],xlist[point]].to_dataframe()['sfcelv']
    rmdtrnd_pt = rmdtrnd.rmdtrnd[:,ylist[point],xlist[point]].to_dataframe()['rmdtrnd']
    rmdsesn_pt = rmdsesn.rmdsesn[:,ylist[point],xlist[point]].to_dataframe()['rmdsesn']
    standardized_pt = standardized.standardize[:,ylist[point],xlist[point]].to_dataframe()['standardize']
    
    sfcelv_pt.plot(ax=ax[0],color='k')
    rmdtrnd_pt.plot(ax=ax[1],color='r')
    rmdsesn_pt.plot(ax=ax[2],color='b')
    standardized_pt.plot(ax=ax[3],color='m')
    
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('WSE (m)')
    ax[1].set_ylabel('Trend \n Removed')
    ax[2].set_ylabel('Seasonality \n Removed')
    ax[3].set_ylabel('Standardized')
    
    pathname=os.path.join('p01_timeseries',rivername)
    mk_dir(pathname)
    figname = os.path.join(pathname,pname[point]+'_timeseries.png')
    plt.tight_layout()
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    
    
    