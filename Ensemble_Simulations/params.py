# uncompyle6 version 3.9.0
# Python bytecode version base 2.7 (62211)
# Decompiled from: Python 3.8.2 (default, Mar  8 2020, 15:23:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-28)]
# Embedded file name: params.py
# Compiled at: 2023-04-27 02:19:50
import os

########################
#
# parameters list
#
########################
# 1. experimental settings
def expname():
    # return "ECMWF049_050"
    # return "ECMWF049_150"
    # return "AMZ049"
    # return "AMZCAL049"
    return "AMZ" #amazone
    # return "AMZCAL"
    # return "GLB" #global
    # return "CONUS" #conus

def mapname():
    return "amz_06min"
    # return "glb_15min"
    # return "conus_06min"

# 2. time settings
def timestep():
    return 86400 # outer timestep in seconds

def starttime():
    return (2000,1,1) # start date: [year,month,date]
    #return (2015,1,1)

def endtime():
    return (2021,1,1) # end date: [year,month,date]
                      # *note: this date is not included
    #return (2016,1,1)

def start_year():
    return 2000
    #return 2015

def end_year():
    return 2020
    #return 2016

# 3. input runoff forcing
def runname():
    return "ERA5"
    # return "isimip3a"

def rundir():
    # return "/work/a04/julien/CaMa-Flood_v4/inp/isimip3a/runoff" #isimip3a
    # return "/work/a02/menaka/ERA5/bin" #ERA5
    return os.path.join(inputdir(), runname(), 'bin')

def inputdir():
    return os.path.join(org_dir(), 'CaMa_in')
    # return"/work/a06/menaka/ensemble_simulations/CaMa_in"
    # return "/cluster/data6/menaka/HydroDA/inp"
    #return "/home/yamadai/data/Runoff/E2O/nc"
    #return "/cluster/data6/menaka/ensemble_org/CaMa_in/VIC_BC/Roff"
    # return "/cluster/data6/menaka/covariance/CaMa_in/VIC/Roff"
    # return "/cluster/data6/menaka/HydroDA/inp/"
    # return "/work/a02/menaka/"
    # return "./CaMa_in/"

def input():
    # return "ECMWF"
    # return "E2O"
    return "ERA5"
    # return "isimip3a"

# 4. parameters for pertubating runoff
def method():
    # return "simple"
    # return "normal"
    return 'lognormal'

def distopen():
    return 0.5 # not needed for ERA20CM
    # corrupted runoff's percentage
    # 0.75 for original Data Assimilation simulation (25% reduced)
    # 1.25 for 25% increased simulation
    # 1.00 for simulation using 1 year before runoff
    # *note: also editing and and re-compile of control_inp at CaMa-Flood is nessessary

def diststd():
    return 0.25 # not needed for ERA20CM
    # noise to make runoff input to scatter ensembles

def beta():
    return 0.0

def alpha():
    return 0.993
    #return 1.0 - (1.0/150.0)

def E():
    # return 50.0
    return 0.30

# 5. ensemble members
def ens_mem():
    return 20
    # number of ensemble members

def mode():
    return 4
    # parameter to change assimilation mode
    # 1: Earth2Obs, 2: ERA20CM, 3: -25% ELSE_KIM2009, 4: ERA5, 5: ECMWF

def run_flag():
    return 0
    # 0 run all simulations
    # 1 run only corrupted and assimilated simulations
    # 2 run only true and assimilated simulations
    # 3 run only assimilated simulation

# 6. CaMa-Flood related settings
def spinup_flag():
    return 0
    # 1: no spinup simulation simulation 
    # 0: do spinup simulation simulation

def CaMa_dir():
    return '/cluster/data8/abdul.moiz/20230511_CaMa-DA/CaMa-Flood_v4.1'
    # directory of CaMa-Flood
    # indicate the directory of ./map or ./src and other folders

def org_dir():
    return '/cluster/data8/abdul.moiz/20230511_CaMa-DA/Ensemble_Simulations'
    # return "/cluster/data7/menaka/ensemble_simulations"

def para_nums():
    return 5
    # setting number of parallels to run CaMa-Flood Model
    # defualt is 6, but may change depending on your system

def cpu_nums():
    return 8
    # number of cpus used 

def version():
    return 'v1.0.0 (updated 2021-06-17)\n Simulating Ensembles'
# okay decompiling params.pyc

# 7. HydroDA Directory
def hydro_da_dir():
    return '/cluster/data8/abdul.moiz/20230511_CaMa-DA/HydroDA'


