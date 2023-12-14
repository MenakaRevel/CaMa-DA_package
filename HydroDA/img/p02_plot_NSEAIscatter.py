import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
import glob
import calendar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import multiprocessing
from functools import partial
from collections import Counter
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import re

def initialize(outpath):
    sys.path.append(outpath)
    import params as pm
    return pm

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

def txt_vector(west,east,north,south,camadir,mapname):
    tmpfile1 = 'NSEAItmp1.txt'
    box = "%f %f %f %f"%(west,east,north,south) 
    os.system("./bin/txt_vector "+box+" "+camadir+" "+mapname+" > "+tmpfile1) 
    return tmpfile1
    
def rivvec(level,west,east,north,south,camadir,mapname):
    tmpfile1 = txt_vector(west,east,north,south,camadir,mapname)
    tmpfile2 = "NSEAItmp_%02d.txt"%(level)
    os.system("./bin/print_rivvec "+tmpfile1+" 1 "+str(level)+" > "+tmpfile2)
    rivvec_df = pd.read_csv(tmpfile2,delim_whitespace=True,header=None,names=['lon1','lat1','uparea','lon2','lat2'])
    return rivvec_df
   
def read_grdc_data_id(GRDC_dir,GRDC_ID):
    #GRDC Parameters
    grdc_params = pd.read_csv(os.path.join(GRDC_dir,'GRDC_Stations.csv'),encoding='cp1252',index_col='grdc_no')
    grdc_params = grdc_params.loc[int(GRDC_ID),:]
    
    #GRDC Data
    grdc_data = pd.read_csv(os.path.join(GRDC_dir,GRDC_ID+'_Q_Day.Cmd.txt'),skiprows=37,encoding='cp1252',header=None,delim_whitespace=True)
    grdc_data.columns=['Date','Discharge (m$^3$/s)']
    grdc_data['Date'] = grdc_data['Date'].str.replace(';--:--;','')
    grdc_data['Date'] = grdc_data['Date'].astype('datetime64[ns]')
    grdc_data.index = grdc_data['Date']
    grdc_data.index.name = 'Date'
    grdc_data.drop(columns=['Date'],inplace=True)
    
    return grdc_params,grdc_data

def read_grdc_data_rivername(GRDC_dir,rivername):
    #GRDC Parameters
    grdc_params = pd.read_csv(os.path.join(GRDC_dir,'GRDC_Stations.csv'),encoding='cp1252',index_col='grdc_no')
    grdc_params = grdc_params[grdc_params['river'].isin(rivernames)]
    
    grdc_id_filter_nan = []
    grdc_data_filter_nan = []
    
    for grdc_id in grdc_params.index:
        try:
            tmp, grdc_data = read_grdc_data_id(GRDC_dir,str(grdc_id))
            grdc_data.columns = [grdc_id]
            grdc_id_filter_nan.append(grdc_id)
            grdc_data_filter_nan.append(grdc_data)
        except:
            pass
    grdc_id_filter_nan = grdc_params[(grdc_params.index).isin(grdc_id_filter_nan)]
    grdc_data_filter_nan = pd.concat(grdc_data_filter_nan,axis=1)
    grdc_data_filter_nan[grdc_data_filter_nan<0] = 0 # Replace negative values with zero
    return grdc_id_filter_nan,grdc_data_filter_nan

def read_grdc_loc(grdc_params,grdc_loc):
    grdc_loc = pd.read_csv(grdc_loc,header=0,sep=';',index_col=0)
    grdc_loc = grdc_loc.loc[(grdc_loc.index).isin(grdc_params.index)]   #This Needs to be changed # Confirm with Menaka
    grdc_loc.columns = grdc_loc.columns.str.strip() # Remove whitespace from columns
    grdc_loc = grdc_loc.iloc[:,:-1]
    grdc_params = grdc_params.loc[grdc_loc.index]
    grdc_params = pd.concat([grdc_params,grdc_loc],axis=1)
    return grdc_params

def filter_nan(s,o):
    """
    this functions removed the data  from simulated and observed data
    where ever the observed data contains nan
    """
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]

    return data[:,0],data[:,1]
    
def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    print(s,o)
    df = pd.concat([s,o],axis=1)
    df.columns = ['Simulated','Observed']
    df = df.dropna()
    if df.empty:
        return np.nan
    s=df['Simulated']
    o=df['Observed']
    # s,o = filter_nan(s,o)
    # o=np.ma.masked_where(o<=0.0,o).filled(0.0)
    # s=np.ma.masked_where(o<=0.0,s).filled(0.0)
    # o=np.compress(o>0.0,o)
    # s=np.compress(o>0.0,s) 
    return 1 - sum((s-o)**2)/(sum((o-np.mean(o))**2)+1e-20)


# def makedir(path):
#     os.makedirs(path,exist_ok=True)
#     return path

# def read_grdc_data(GRDC_dir,GRDC_ID):
#     #GRDC Parameters
#     grdc_params = pd.read_csv(os.path.join(GRDC_dir,'GRDC_Stations.csv'),encoding='cp1252',index_col='grdc_no')
#     grdc_params = grdc_params.loc[int(GRDC_ID),:]
    
#     #GRDC Data
#     grdc_data = pd.read_csv(os.path.join(GRDC_dir,GRDC_ID+'_Q_Day.Cmd.txt'),skiprows=37,encoding='cp1252',header=None,delim_whitespace=True)
#     grdc_data.columns=['Date','Discharge (m$^3$/s)']
#     grdc_data['Date'] = grdc_data['Date'].str.replace(';--:--;','')
#     grdc_data['Date'] = grdc_data['Date'].astype('datetime64[ns]')
#     grdc_data.index = grdc_data['Date']
#     grdc_data.index.name = 'Date'
#     grdc_data.drop(columns=['Date'],inplace=True)
    
#     return grdc_params,grdc_data

def read_parallel_ens_data(out_dir,ensembles,nx,ny,x_coord,y_coord):
    df_ens = []
    for ensemble in range(1,ensembles+1):
        files = sorted(glob.glob(os.path.join(out_dir,'*_'+str(ensemble).zfill(3)+'.bin')))
        final_counter = Counter()
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for result in pool.imap_unordered(partial(read_bin_f_point,
                                                      nx=nx,
                                                      ny=ny,
                                                      x_coord=x_coord,
                                                      y_coord=y_coord), 
                                              files):
                    final_counter.update(result)
        df = pd.DataFrame.from_dict(final_counter,orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = [ensemble]
        df_ens.append(df)
    df_ens = pd.concat(df_ens,axis=1)
    return df_ens

def read_bin_f_point(fname,
                      nx,ny,
                      x_coord,y_coord):
    #print(fname,nx,ny,x_coord,y_coord)
    c = Counter()
    bin_f = np.fromfile(fname,np.float32).reshape(ny,nx)
    bin_f = bin_f[y_coord-1,x_coord-1]
    label = fname.split('/')[-1]
    date = label.split('_')[0][-8:]
    ensemble = label.split('_')[-1].split('.')[0]
    print(date,ensemble)
    c[date]=bin_f
    return c

def read_parallel_ens_multipointdata(out_dir,ensembles,nx,ny,grdc_params):
    df_ens_stations = {}
    df_ens = []
    for ensemble in range(1,ensembles+1):
        files = sorted(glob.glob(os.path.join(out_dir,'*_'+str(ensemble).zfill(3)+'.bin')))
        final_counter = Counter()
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for result in pool.imap_unordered(partial(read_bin_f_multipoint,
                                                      nx=nx,
                                                      ny=ny,
                                                      grdc_params=grdc_params), 
                                              files):
                    final_counter.update(result)
        df = pd.DataFrame.from_dict(final_counter,orient='index')
        df.columns = [ensemble]
        df_ens.append(df)
    df_ens = pd.concat(df_ens,axis=1)
    
    # Separating DataFrame for Each GRDC Station
    for grdc_id in grdc_params.index:
        df_station = df_ens.xs(grdc_id, level=1, drop_level=True)
        df_station.index = pd.to_datetime(df_station.index)
        df_station = df_station.sort_index()
        df_ens_stations[grdc_id] = df_station
    return df_ens_stations

def read_bin_f_multipoint(fname,
                      nx,ny,
                      grdc_params):
    #print(fname,nx,ny,x_coord,y_coord)
    c = Counter()
    for grdc_id in grdc_params.index:
        x_coord = grdc_params.loc[grdc_id,'ix1']
        y_coord = grdc_params.loc[grdc_id,'iy1']
        bin_f = np.fromfile(fname,np.float32).reshape(ny,nx)
        bin_f = bin_f[y_coord-1,x_coord-1]
        label = fname.split('/')[-1]
        date = label.split('_')[0][-8:]
        ensemble = label.split('_')[-1].split('.')[0]
        print(date,ensemble)
        c[date,grdc_id]=bin_f
    return c

# def read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,nx,ny,x_coord,y_coord,varname):
#     df_sims = []
#     for year in range(start_year,end_year+1):
#         df_sim = pd.DataFrame(index=pd.date_range(str(year)+'-01-01',str(year)+'-12-31',freq='D'),columns=['Control'])
#         print(year)
#         if calendar.isleap(year) == True:
#             timesteps = 366
#         elif calendar.isleap(year) == False:
#             timesteps = 365
#         fname = varname+str(year)+'.bin'
#         var_bin = np.fromfile(os.path.join(CaMa_out_ctrl_dir,fname),np.float32).reshape(timesteps,ny,nx)
#         df_sim.loc[:,'Control'] = var_bin[:,y_coord-1,x_coord-1]
#         df_sims.append(df_sim)
#     df_sims = pd.concat(df_sims)
#     return df_sims

# def read_CaMa_out_ens_obs_point(CaMa_out_ens_dir,start_year,end_year,nx,ny,x_coord,y_coord,varname,prefix):
#     df_sim_ens = []
#     for ensemble in range(1,ensembles+1):
#         prefix_ens = prefix+str(ensemble).zfill(3)
#         print('Ensemble: ',ensemble)
#         df_sim = read_CaMa_out_obs_point(os.path.join(CaMa_out_ens_dir,'CaMa_out',prefix_ens),start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)
#         df_sim = df_sim.rename(columns={'Control':ensemble})
#         df_sim_ens.append(df_sim)
#     df_sim_ens = pd.concat(df_sim_ens,axis=1)
#     return df_sim_ens

# # Data Directories
# assim_out_ens_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/HydroDA/out/test_wse/assim_out/outflw'
# assim_outputs = ['assim','open']
# assim_outputs_c = ['r','b']
# GRDC_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/GRDC_2019'
# CaMa_out_ctrl_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Empirical_LocalPatch/CaMa_out/amz_06min_ERA5'
# fig_name = 'test_p01.png'

# # GRDC Station ID
# GRDC_ID = '3629001'

# # Data Properties
# glob_nx=3600
# glob_ny=1800

# reg_nx=350
# reg_ny=250

# # Selected Point Coordinates
# glob_x_coord = 1245
# glob_y_coord = 920

# reg_x_coord = 245
# reg_y_coord = 70

# start_year=2014
# end_year=2015

# Number of Ensembles
ensembles=20

# varname = 'outflw'
# varunits = 'm^3/s'

# # Read GRDC Data
# grdc_params,grdc_data =  read_grdc_data(GRDC_dir, GRDC_ID)

# # Read Control Simulation
# ERA5_discharge = read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)


# # Plotting Figure
# fig,ax = plt.subplots(figsize=(8,5))
# i=0
# for assim_output in assim_outputs:
#     df = read_parallel_ens_data(os.path.join(assim_out_ens_dir,assim_output),
#                                 ensembles,
#                                 reg_nx,reg_ny,
#                                 reg_x_coord,reg_y_coord)
#     df.name = assim_output
#     df.plot(ax=ax,color=assim_outputs_c[i],legend=False,alpha=0.5,lw=0.5)
#     df.mean(axis=1).plot(label=assim_output,color=assim_outputs_c[i],legend=True)
#     i+=1
# grdc_data.loc[df.index,:].plot(y=grdc_data.iloc[:,-1:].columns[0],color='g',legend=True,ax=ax,label='GRDC')
# ERA5_discharge.loc[df.index,:].plot(y=ERA5_discharge.iloc[:,-1:].columns[0],color='k',legend=True,ax=ax,label='Control')
# plt.tight_layout()
# plt.title(grdc_params['station'])
# ax.set_ylabel(varname+' '+'($'+varunits+'$)')
# fig.savefig(fig_name,dpi=300,bbox_inches='tight')


#assim_out = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/HydroDA/out/test_wse'
#assim_out = '/cluster/data6/menaka/HydroDA/out/test_wse'
expname='test_wse'
assim_out = '../HydroDA/out/'+test_wse
assim_out_open = os.path.join(assim_out,'assim_out','outflw','open')
assim_out_assim = os.path.join(assim_out,'assim_out','outflw','assim')
w=0.15*2
GRDC_dir = '../../GRDC'
GRDC_loc = '/cluster/data6/menaka/CaMa-Flood_v4/map/amz_06min/grdc_loc.txt' # CaMa-Flood lookup table
rivernames  = ["AMAZON RIVER","AMAZON","NEGRO, RIO","PURUS, RIO","MADEIRA, RIO","JURUA, RIO"
              ,"TAPAJOS, RIO","XINGU, RIO","CURUA, RIO","JAPURA, RIO","BRANCO, RIO"
              ,"JAVARI, RIO","IRIRI, RIO","JURUENA, RIO","ACRE, RIO","BENI, RIO"
              ,"MAMORE, RIO","GUAPORE, RIO","ARINOS, RIO","TROMBETAS, RIO"]


cmap='viridis'
vmin=-1.0
vmax=1.0
norm=Normalize(vmin=vmin,vmax=vmax)


# Initialize (Read params.py)
pm = initialize(assim_out)

# Read Regional Map Parameters
nx,ny,gsize,west,east,south,north=read_params(os.path.join(pm.CaMa_dir(),'map',pm.mapname(),'params.txt'))


#Calculating NSEAI
grdc_params,grdc_data = read_grdc_data_rivername(GRDC_dir,rivernames)
grdc_params = read_grdc_loc(grdc_params,GRDC_loc)
print(grdc_data)
assim_stations=read_parallel_ens_multipointdata(assim_out_assim,ensembles,nx,ny,grdc_params)
open_stations=read_parallel_ens_multipointdata(assim_out_open,ensembles,nx,ny,grdc_params)

df_stations_nseai = pd.DataFrame(index=grdc_params.index.tolist(),columns=['NS_open','NS_assim','NSEAI'])
print(df_stations_nseai)  

for grdc_id in grdc_params.index:
    df_assim = assim_stations[grdc_id].mean(axis=1)
    df_open = open_stations[grdc_id].mean(axis=1)
    df_obs = grdc_data.loc[:,grdc_id]
    
    NS_assim = NS(df_assim,df_obs)
    df_stations_nseai.loc[grdc_id,'NS_assim'] = NS_assim
    
    NS_open = NS(df_open,df_obs)
    df_stations_nseai.loc[grdc_id,'NS_open'] = NS_open
    
    NSEAI=(NS_assim-NS_open)/((1.0-NS_open)+1.0e-20)
    df_stations_nseai.loc[grdc_id,'NSEAI'] = NSEAI
    
df_stations_nseai = pd.concat([df_stations_nseai,grdc_params[['station','lat','long']]],axis=1)

    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_stations_nseai)
    

# Plotting Background
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())

ax.set_extent([west, east, south+1, north])

for level in range(1,10+1):
    rivvec_df = rivvec(level,west,east,north,south,pm.CaMa_dir(),pm.mapname())
    width=float(level)*w
    for i in rivvec_df.index:
        ax.plot([rivvec_df.loc[i,'lon1'],rivvec_df.loc[i,'lon2']],
                    [rivvec_df.loc[i,'lat1'],rivvec_df.loc[i,'lat2']],
                    linewidth=width,color='w')


ax.add_feature(cartopy.feature.LAND,facecolor='lightgray')
ax.add_feature(cartopy.feature.COASTLINE,zorder=3)


#Plotting Points
im=plt.scatter(x=df_stations_nseai['long'],y=df_stations_nseai['lat'],
            c=df_stations_nseai['NSEAI'],
            transform=ccrs.PlateCarree(),zorder=3,
            cmap=cmap,
            norm=norm)
cb=plt.colorbar(im)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='k', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.tight_layout()
fig.savefig('p02_NSEAIscatter.png',dpi=300,bbox_inches='tight')
                            