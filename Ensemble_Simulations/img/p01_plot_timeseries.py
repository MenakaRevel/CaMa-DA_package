import pandas as pd
import xarray as xr
import numpy as np
import os
import glob
import calendar
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from collections import Counter

def makedir(path):
    os.makedirs(path,exist_ok=True)
    return path

def read_grdc_data(GRDC_dir,GRDC_ID):
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

def read_parallel_era5_data(out_dir,nx,ny,x_coord,y_coord,start_year,end_year):
    df_year = []
    for year in range(start_year,end_year+1):
        files = sorted(glob.glob(os.path.join(out_dir,'Roff____'+str(year)+'*.sixmin')))
        final_counter = Counter()
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for result in pool.imap_unordered(partial(read_era5_bin_f_point,
                                                      nx=nx,
                                                      ny=ny,
                                                      x_coord=x_coord,
                                                      y_coord=y_coord), 
                                              files):
                    final_counter.update(result)
        df = pd.DataFrame.from_dict(final_counter,orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['ERA5']
        df_year.append(df)
    df_year = pd.concat(df_year,axis=0)
    df_year = df_year.sort_index()
    return df_year

def read_era5_bin_f_point(fname,
                     nx,ny,
                     x_coord,y_coord):
    #print(fname,nx,ny,x_coord,y_coord)
    c = Counter()
    bin_f = np.fromfile(fname,np.float32).reshape(ny,nx)
    bin_f = bin_f[y_coord-1,x_coord-1]
    label = fname.split('/')[-1]
    date = label.split('_')[-1].split('.')[0]
    print(date)
    c[date]=bin_f
    return c

def read_parallel_ens_roff_data(out_dir,ensembles,nx,ny,x_coord,y_coord,start_year,end_year):
    df_ens = []
    for ensemble in range(1,ensembles+1):
        df_year = []
        for year in range(start_year,end_year+1):
            files = sorted(glob.glob(os.path.join(out_dir,'Roff__'+str(year)+'*'+str(ensemble).zfill(3)+'.one')))
            final_counter = Counter()
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                for result in pool.imap_unordered(partial(read_roff_bin_f_point,
                                                          nx=nx,
                                                          ny=ny,
                                                          x_coord=x_coord,
                                                          y_coord=y_coord), 
                                                  files):
                        final_counter.update(result)
            df = pd.DataFrame.from_dict(final_counter,orient='index')
            df.index = pd.to_datetime(df.index)
            df_year.append(df)
        df_year = pd.concat(df_year,axis=0)
        df_year.columns = [ensemble]
        df_ens.append(df_year)
    df_ens = pd.concat(df_ens,axis=1)
    return df_ens

def read_roff_bin_f_point(fname,
                     nx,ny,
                     x_coord,y_coord):
    #print(fname,nx,ny,x_coord,y_coord)
    c = Counter()
    bin_f = np.fromfile(fname,np.float32).reshape(ny,nx)
    bin_f = bin_f[y_coord-1,x_coord-1]
    label = fname.split('/')[-1]
    date = label.split('_')[-1].split('.')[0]
    ensemble = date[8:]
    date = date[0:8]
    print(date,ensemble)
    c[date]=bin_f
    return c

def read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,nx,ny,x_coord,y_coord,varname):
    df_sims = []
    for year in range(start_year,end_year+1):
        df_sim = pd.DataFrame(index=pd.date_range(str(year)+'-01-01',str(year)+'-12-31',freq='D'),columns=['Control'])
        print(year)
        if calendar.isleap(year) == True:
            timesteps = 366
        elif calendar.isleap(year) == False:
            timesteps = 365
        fname = varname+str(year)+'.bin'
        var_bin = np.fromfile(os.path.join(CaMa_out_ctrl_dir,fname),np.float32).reshape(timesteps,ny,nx)
        df_sim.loc[:,'Control'] = var_bin[:,y_coord-1,x_coord-1]
        df_sims.append(df_sim)
    df_sims = pd.concat(df_sims)
    return df_sims

def read_CaMa_out_ens_obs_point(CaMa_out_ens_dir,start_year,end_year,nx,ny,x_coord,y_coord,varname,prefix):
    df_sim_ens = []
    for ensemble in range(1,ensembles+1):
        prefix_ens = prefix+str(ensemble).zfill(3)
        print('Ensemble: ',ensemble)
        df_sim = read_CaMa_out_obs_point(os.path.join(CaMa_out_ens_dir,'CaMa_out',prefix_ens),start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)
        df_sim = df_sim.rename(columns={'Control':ensemble})
        df_sim_ens.append(df_sim)
    df_sim_ens = pd.concat(df_sim_ens,axis=1)
    return df_sim_ens

# Data Directories
ERA5_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Ensemble_Simulations/CaMa_in/ERA5/bin'
roff_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Ensemble_Simulations/CaMa_in/ERA5/Roff'
CaMa_out_ctrl_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Empirical_LocalPatch/CaMa_out/amz_06min_ERA5'
CaMa_out_ens_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Ensemble_Simulations'
GRDC_dir = '/cluster/data8/abdul.moiz/20230511_CaMa-DA/GRDC_2019'


# GRDC Station ID
GRDC_ID = '3629001'

# Data Properties

glob_nx=3600
glob_ny=1800

reg_nx=350
reg_ny=250

# Selected Point Coordinates
glob_x_coord = 1245
glob_y_coord = 920

reg_x_coord = 245
reg_y_coord = 70


# Selected Time Period
start_year = 2014
end_year = 2015

# Number of Ensembles
ensembles=20

# CaMa out
varname = 'outflw'
prefix = 'AMZERA5'

# Read GRDC Data
grdc_params,grdc_data =  read_grdc_data(GRDC_dir, GRDC_ID)


#Plot Runoff
fig, ax = plt.subplots(figsize=(8,5))

ERA5_roff = read_parallel_era5_data(ERA5_dir,
                            glob_nx,glob_ny,
                            glob_x_coord,glob_y_coord,
                            start_year,end_year)

Ensemble_roff = read_parallel_ens_roff_data(roff_dir,ensembles,
                                glob_nx,glob_ny,
                                glob_x_coord,glob_y_coord,
                                start_year,end_year)
Ensemble_roff.iloc[:,:-1].plot(color='r', alpha=0.3,legend=False,ax=ax)
Ensemble_roff.iloc[:,-1:].plot(y=Ensemble_roff.iloc[:,-1:].columns[0],color='r', alpha=0.3,legend=True,ax=ax,label='Ensemble')
ERA5_roff.plot(y=ERA5_roff.iloc[:,-1:].columns[0],color='k',legend=True,ax=ax,label='ERA5')
ax.set_ylabel('Runoff (mm)',fontsize=12)
ax.set_title('Location: '+grdc_params['station'],fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join('test_p01a.png'),dpi=300,bbox_inches='tight')

# Plot Outflow (River Discharge)
varname = 'outflw'
fig, ax = plt.subplots(figsize=(8,5))
ERA5_discharge = read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)
Ensemble_discharge = read_CaMa_out_ens_obs_point(CaMa_out_ens_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname,prefix)

Ensemble_discharge.iloc[:,:-1].plot(color='r', alpha=0.3,legend=False,ax=ax)
Ensemble_discharge.iloc[:,-1:].plot(y=Ensemble_discharge.iloc[:,-1:].columns[0],color='r', alpha=0.3,legend=True,ax=ax,label='Ensemble')
ERA5_discharge.plot(y=ERA5_discharge.iloc[:,-1:].columns[0],color='k',legend=True,ax=ax,label='Control')
grdc_data.loc[ERA5_discharge.index,:].plot(y=grdc_data.iloc[:,-1:].columns[0],color='b',legend=True,ax=ax,label='GRDC')
ax.set_ylabel('River Discharge (m$^3$/s)',fontsize=12)
ax.set_title('Location: '+grdc_params['station'],fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join('test_p01b.png'),dpi=300,bbox_inches='tight')

# Plot WSE (m)
varname = 'sfcelv'
fig, ax = plt.subplots(figsize=(8,5))
ERA5_wse = read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)
Ensemble_wse = read_CaMa_out_ens_obs_point(CaMa_out_ens_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname,prefix)

Ensemble_wse.iloc[:,:-1].plot(color='r', alpha=0.3,legend=False,ax=ax)
Ensemble_wse.iloc[:,-1:].plot(y=Ensemble_wse.iloc[:,-1:].columns[0],color='r', alpha=0.3,legend=True,ax=ax,label='Ensemble')
ERA5_wse.plot(y=ERA5_wse.columns[0],color='k',legend=True,ax=ax,label='Control')
ax.set_ylabel('WSE (m)',fontsize=12)
ax.set_title('Location: '+grdc_params['station'],fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join('test_p01c.png'),dpi=300,bbox_inches='tight')


    





