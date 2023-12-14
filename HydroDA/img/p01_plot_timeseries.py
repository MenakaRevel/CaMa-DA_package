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
expname='test_wse'
cltname='amz_06min_ERA5'
assim_out_ens_dir = '../out/'+expname+'/assim_out/outflw' # assimilated outflw
assim_outputs = ['assim','open']
assim_outputs_c = ['r','b']
GRDC_dir = '../../GRDC' # observation
CaMa_out_ctrl_dir = '../../Empirical_LocalPatch/CaMa_out/'+cltname # control simulation
fig_name = 'test_p01.png'

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

start_year=2014
end_year=2015

# Number of Ensembles
ensembles=20

varname = 'outflw'
varunits = 'm^3/s'

# Read GRDC Data
grdc_params,grdc_data =  read_grdc_data(GRDC_dir, GRDC_ID)

# Read Control Simulation
ERA5_discharge = read_CaMa_out_obs_point(CaMa_out_ctrl_dir,start_year,end_year,reg_nx,reg_ny,reg_x_coord,reg_y_coord,varname)


# Plotting Figure
fig,ax = plt.subplots(figsize=(8,5))
i=0
for assim_output in assim_outputs:
    df = read_parallel_ens_data(os.path.join(assim_out_ens_dir,assim_output),
                                ensembles,
                                reg_nx,reg_ny,
                                reg_x_coord,reg_y_coord)
    df.name = assim_output
    df.plot(ax=ax,color=assim_outputs_c[i],legend=False,alpha=0.5,lw=0.5)
    df.mean(axis=1).plot(label=assim_output,color=assim_outputs_c[i],legend=True)
    i+=1
grdc_data.loc[df.index,:].plot(y=grdc_data.iloc[:,-1:].columns[0],color='g',legend=True,ax=ax,label='GRDC')
ERA5_discharge.loc[df.index,:].plot(y=ERA5_discharge.iloc[:,-1:].columns[0],color='k',legend=True,ax=ax,label='Control')
plt.tight_layout()
plt.title(grdc_params['station'])
ax.set_ylabel(varname+' '+'($'+varunits+'$)')
fig.savefig(fig_name,dpi=300,bbox_inches='tight')