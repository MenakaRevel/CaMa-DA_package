#!/bin/sh
#====================
# create statitcs (e.g., mean, std) from netCDF4 files
# dimesnion (nx, ny)
# Menaka@IIS
# 2020/05/29
#===========================
#*** PBS setting when needed
#PBS -q F20
#PBS -l select=1:ncpus=20:mem=10gb
#PBS -j oe
#PBS -m ea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N copy2HydroDA
#===========================
# cd $PBS_O_WORKDIR


# input settings
syear=`python -c "import params; print (params.starttime()[0])"`
smonth=`python -c "import params; print (params.starttime()[1])"`
sdate=`python -c "import params; print (params.starttime()[2])"`
eyear=`python -c "import params; print (params.endtime()[0])"`
emonth=`python -c "import params; print (params.endtime()[1])"`
edate=`python -c "import params; print (params.endtime()[2])"`
# for reading bin
ssyear=`python -c "import params; print (params.start_year())"`
eeyear=`python -c "import params; print (params.end_year())"`
echo $ssyear" to "$eeyear
# names
CAMADIR=`python -c "import params; print (params.CaMa_dir())"`
outdir="./" #`python -c "import params; print (params.out_dir())"`
cpunums=`python -c "import params; print (params.cpu_nums())"`
mapname=`python -c "import params; print (params.mapname())"`
expname=`python -c "import params; print (params.expname())"`
runname=`python -c "import params; print (params.runname())"`
ens_num=`python -c "import params; print (params.ens_mem())"`
#=================================================
# output directory -> HydroDA/dat
odir=`python -c "import params; print (params.hydro_da_dir())"`

#=================================================

# Make HydroDA Data Directory
odir_dat=$odir/dat
mkdir -p $odir_dat

#Setting Prefix
prefix=sfcelv_${ens_num}

# Ensemble Loop
START=1
END=$((ens_num))
echo $END
for (( ens=$START; ens<=$END; ens++ ))
do
    ens_char=$(printf "%03d" $ens)
    echo $ens_char

    # Mean
    infile=./CaMa_out/${expname}${runname}${ens_char}/sfcelv_mean${ssyear}-${eeyear}.bin
    outfile=${odir_dat}/mean_${prefix}_${runname}_${mapname}_${ssyear}-${eeyear}_${ens_char}.bin
    echo $infile
    echo $outfile
    cp -vR $infile $outfile

    #STD
    infile=./CaMa_out/${expname}${runname}${ens_char}/sfcelv_std${ssyear}-${eeyear}.bin
    outfile=${odir_dat}/std_${prefix}_${runname}_${mapname}_${ssyear}-${eeyear}_${ens_char}.bin
    echo $infile
    echo $outfile
    cp -vR $infile $outfile

done


