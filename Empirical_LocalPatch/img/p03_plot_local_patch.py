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
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import re

#Custom Fuctions
sys.path.append('../')
import params as pm
import read_grdc


def txt_vector(west,east,north,south,CaMa_dir,mapname):
    box="%f %f %f %f"%(west,east,north,south)
    os.system("./bin/txt_vector "+box+" "+CaMa_dir+" "+mapname+" > tmp.txt")
    df = pd.read_csv('tmp.txt',header=None,delim_whitespace=True)
    df.columns = ['lon1','lat1','lon2','lat2','uparea']
    return df

def rivvec(LEVEL):
    os.system("./bin/print_rivvec tmp.txt 1 "+str(LEVEL)+" > tmp2.txt")
    df = pd.read_csv('tmp2.txt',delim_whitespace=True,header=None,names=['lon1','lat1','uparea','lon2','lat2'])
    return df

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

# Override
print(pm.out_dir())

rivername = 'AMAZONAS'
lw=0.3
cmap = 'viridis_r'
vmin = 0.0
vmax = 1.0

# Read Regional Map Parameters
nx,ny,gsize,west,east,south,north=read_params(os.path.join(pm.CaMa_dir(),'map',pm.map_name(),'params.txt'))

# Regional map dimesion relative to global map
offsetX = int((west+180.0)/gsize)
offsetY = int((90.0-north)/gsize)

# Global Dimension
glbname="glb_"+pm.map_name()[-5::]
nXX = int((180.0+180.0)/gsize)
nYY = int((90.0+90.0)/gsize)

# Read Regional Maps
nextx,nexty,rivwth,rivhgt,rivlen,elevtn,uparea,lonlat=read_regional_maps(os.path.join(pm.CaMa_dir(),'map',pm.map_name()),nx,ny)
rivermap=((nextx>0))*1.0

# Read rivnum(?)
rivnum = os.path.join(pm.out_dir(),'dat','rivnum_'+pm.map_name()[-5::]+'.bin')
rivnum = np.fromfile(rivnum,np.int32).reshape(nYY,nXX)


# Read Higher Resolution Data
catmxy = pm.CaMa_dir()+"/map/"+pm.map_name()+"/1min/1min.catmxy.bin"
catmxy_ctl = pd.read_csv(catmxy.replace('.bin','.ctl'), skiprows = lambda x: x not in [4,5],delim_whitespace=True,header=None,index_col=0)
catmxy = np.fromfile(catmxy,np.int16).reshape(2,catmxy_ctl.loc['ydef',1],catmxy_ctl.loc['xdef',1])
catmx = catmxy[0]
catmy = catmxy[1]

# Read Threshold
threshold = pm.threshold()
damrep = pm.dam_rep()

# Patchname
if damrep == 1:
  patchname=pm.map_name()+'_'+pm.input_name()+'_dam'                      #'amz_06min_ERA5_dam'
  patch_id=pm.map_name()+'_'+pm.input_name()+'_'+str(int(threshold*100))+'_dam' #'amz_06min_ERA5_60_dam'
  #---
  local_patch="local_patch"#"_%3.2f"%(pm.threshold())
  #local_patch1="local_patch_one_%02d_dam"%(threshold*100)
else:
  patchname=pm.map_name()+'_'+pm.input_name()                      #'amz_06min_ERA5'
  patch_id=pm.map_name()+'_'+pm.input_name()+'_'+str(int(threshold*100)) #'amz_06min_ERA5_60'
  #---
  local_patch="local_patch"#"_%3.2f"%(pm.threshold())
  #local_patch1="local_patch_one_%02d"%(threshold*100)

#patch_id = 'amz_06min_S14FD'

# Major Rivers
rivid = {}
rivid_df = os.path.join(pm.out_dir(),'dat','river30_id.txt')
rivid_df = pd.read_csv(rivid_df,header=None,index_col=0)[1]
for i in rivid_df.index:
    rivid[rivid_df[i]] = i

# Read GRDC
pname=[]
xlist=[]
ylist=[]
river=[]
staid=[]

grdc_id,station_loc,x_list,y_list = read_grdc.get_grdc_loc_v396('AMAZON')
river.append([rivername]*len(station_loc))
staid.append(grdc_id)
pname.append(station_loc)
xlist.append(x_list)
ylist.append(y_list)

river=([flatten for inner in river for flatten in inner])
staid=([flatten for inner in staid for flatten in inner])
pname=([flatten for inner in pname for flatten in inner])
xlist=([flatten for inner in xlist for flatten in inner])
ylist=([flatten for inner in ylist for flatten in inner])

cmap = mpl.cm.get_cmap(cmap)
norm = mpl.colors.Normalize(vmin,vmax)

pnum=len(pname)
for point in np.arange(pnum):
    print(point)
    
    ix=xlist[point]+1
    iy=ylist[point]+1
    
    
    #Spatial Weights
    wgt=os.path.join(pm.out_dir(),'weightage','%s','%04d%04d.bin')%(patch_id,ix,iy)
    wgt=np.fromfile(wgt,np.float32).reshape(ny,nx)
    
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([west, east, south, north])

    #------------Temp
    fig_wgt = plt.figure(figsize=(10,5))
    ax_wgt = fig_wgt.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax_wgt.coastlines()
    ax_wgt.set_extent([west, east, south, north])
    
    lat_coords = np.arange(south,north,gsize)[::-1]
    lon_coords = np.arange(west,east,gsize)
    da = xr.DataArray(wgt,coords=[lat_coords,lon_coords],
                                  dims=['lat','lon'])
    da = da.where(da!=0.0)
    da.plot(ax=ax_wgt,cmap='viridis_r')
    
    pathname=os.path.join('p03_local_patch_test2',local_patch,rivername)
    mk_dir(pathname)
    figname = os.path.join(pathname,pname[point]+'_wgt_sptl.png')
    fig_wgt.suptitle(pname[point],y=1.0)
    fig_wgt.savefig(figname,dpi=300,bbox_inches='tight')

    #Plot Background River Vector Map
    lllat, urlat, lllon, urlon = latlon_river(rivername,ix,iy)
    txt_vector_df = txt_vector(lllon,urlon,urlat,lllat-1,pm.CaMa_dir(),glbname)
    for LEVEL in range(1,10+1):
        width=float(LEVEL)*lw
        rivvec_df = rivvec(LEVEL)
        #print(rivvec_df)
        for i in rivvec_df.index:
            lon1 = rivvec_df.loc[i,'lon1']
            lon2 = rivvec_df.loc[i,'lon2']
            lat1 = rivvec_df.loc[i,'lat1']
            lat2 = rivvec_df.loc[i,'lat2']
        
            ax.plot([lon1,lon2],[lat1,lat2],color='#C0C0C0',linewidth=width,zorder=101)

    #Plot Variable Vector Map (Weights)
    lllat, urlat, lllon, urlon = latlon_river(rivername,ix,iy)
    txt_vector_df = txt_vector(lllon,urlon,urlat,lllat,pm.CaMa_dir(),glbname)
    for LEVEL in range(1,10+1):
        width=float(LEVEL)*lw
        rivvec_df = rivvec(LEVEL)
        #print(rivvec_df)
        for i in rivvec_df.index:
            lon1 = rivvec_df.loc[i,'lon1']
            lon2 = rivvec_df.loc[i,'lon2']
            lat1 = rivvec_df.loc[i,'lat1']
            lat2 = rivvec_df.loc[i,'lat2']
            
            #- higher resolution data
            ixx1 = int((lon1  - west)*60.0)
            iyy1 = int((-lat1 + north)*60.0)
            
            ix1 =catmxy[0,iyy1,ixx1]- 1
            iy1 =catmxy[1,iyy1,ixx1]- 1
            
            # ixx2 = nextx[ixx1]
            # iyy2 = nexty[iyy1]
            
            if ix1 < 1:
                continue
            
            if rivermap[iy1,ix1] == 0:
                continue
            
            #if wgt[iy1,ix1] <= 0.6:
            #    continue
            
            if lon1-lon2 > 180.0:
                print (lon1, lon2)
                lon2=180.0
            elif lon2-lon1> 180.0:
                print (lon1,lon2)
                lon2=-180.0
            
            colorVal = cmap(norm(wgt[iy1,ix1]))
            #print(lon1,lon2,lat1,lat2,colorVal)
            if wgt[iy1,ix1] >= 0.0:
                ax.plot([lon1,lon2],[lat1,lat2],color=colorVal,linewidth=width,zorder=101)
    
    im=plt.scatter([],[],c=[],cmap=cmap,s=0.1,norm=norm,zorder=101)
    im.set_visible(False)
    l,b,w,h=ax.get_position().bounds
    cax = fig.add_axes([0.82, 0.15, .01, .7])
    cbar=plt.colorbar(im,orientation="vertical",cax=cax)  
    cbar.ax.tick_params(labelsize=10)      
    cbar.set_label('Weights',fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='k', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Plot Target Pixel
    lon = west  + (ix-1)*gsize
    lat = north - (iy-1)*gsize
    ax.scatter(lon,lat,s=75,marker="o",color="red",zorder=150)
    pathname=os.path.join('p03_local_patch_test2',local_patch,rivername)
    mk_dir(pathname)
    figname = os.path.join(pathname,pname[point]+'_wgt.png')
    
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(200)
    #plt.tight_layout()
    fig.suptitle(pname[point],y=1.0)
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    

 
# patch_id = 'amz_06min_S14FD_60'    
# Local Patch Plot
pnum=len(pname)
for point in np.arange(pnum):
    print(point)
    
    ix=xlist[point]+1
    iy=ylist[point]+1
    
    
    # Plot Background River Vector Map
    lllat, urlat, lllon, urlon = latlon_river(rivername,ix,iy)
    txt_vector_df = txt_vector(lllon,urlon,urlat,lllat-1,pm.CaMa_dir(),glbname)
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([west, east, south, north])

    
    for LEVEL in range(1,10+1):
        width=float(LEVEL)*lw
        rivvec_df = rivvec(LEVEL)
        #print(rivvec_df)
        for i in rivvec_df.index:
            lon1 = rivvec_df.loc[i,'lon1']
            lon2 = rivvec_df.loc[i,'lon2']
            lat1 = rivvec_df.loc[i,'lat1']
            lat2 = rivvec_df.loc[i,'lat2']
        
            ax.plot([lon1,lon2],[lat1,lat2],color='#C0C0C0',linewidth=width,zorder=101)
            
    
    # Local Patch
    local_patch_df = pm.out_dir()+"/"+local_patch+"/"+patch_id+"/patch%04d%04d.txt"%(ix,iy)

    with open(local_patch_df,"r") as f:
        lines = f.readlines()
    #--
    with open("tmp.txt","w") as f:
        
        for line in lines:#[:1]:
            line = list(filter(None, re.split(" ",line)))
            #print line
            iix = int(line[0])
            iiy = int(line[1])
            jjx = nextx[iiy-1,iix-1]
            jjy = nexty[iiy-1,iix-1]
            #-----
            lon1 = lonlat[0,iiy-1,iix-1]
            lat1 = lonlat[1,iiy-1,iix-1]
            lon2 = lonlat[0,jjy-1,jjx-1]
            lat2 = lonlat[1,jjy-1,jjx-1]
            #--
            line1="%12.5f %12.5f %12.5f %12.5f %12.1f\n"%(lon1, lat1, lon2, lat2, uparea[iiy-1,iix-1]/1000.**2)
            f.write(line1)
    

    for LEVEL in range(1,10+1):
        os.system("./bin/print_rivvec tmp.txt 1 "+str(LEVEL)+" > tmp2.txt")
        width=float(LEVEL)*lw
        rivvec_df = rivvec(LEVEL)
        
        for i in rivvec_df.index:
            #print(i)
            ax.plot([rivvec_df.loc[i,'lon1'],rivvec_df.loc[i,'lon2']],
                    [rivvec_df.loc[i,'lat1'],rivvec_df.loc[i,'lat2']],
                    linewidth=width,color='b',zorder=102)
            
    
            
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='k', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Target Pixel
    lon = west  + (ix-1)*gsize
    lat = north - (iy-1)*gsize
    plt.scatter(lon,lat,s=75,marker="o",color="red",zorder=105)
    
    pathname=os.path.join('p03_local_patch_test2',local_patch,rivername)
    mk_dir(pathname)
    figname = os.path.join(pathname,pname[point]+'_patch.png')
    
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(200)
    
    fig.suptitle(pname[point],y=1.0)
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    
