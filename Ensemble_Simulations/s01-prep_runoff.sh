#!/bin/sh
#====================
# create statitcs (e.g., mean, std) from netCDF4 files
# dimesnion (nx, ny)
# Menaka@IIS
# 2020/05/29
#===========================
#*** PBS setting when needed
#PBS -q F40
#PBS -l select=1:ncpus=40:mem=60gb
#PBS -j oe
#PBS -m ea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N prep_runoff
#================================================
# cd $PBS_O_WORKDIR
#================================================


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

outdir="./" #`python -c "import params; print (params.out_dir())"`

mapname=`python -c "import params; print (params.mapname())"`
expname=`python -c "import params; print (params.expname())"`

ens_num=`python -c "import params; print (params.ens_mem())"`
runname=`python -c "import params; print (params.runname())"`
rundir=`python -c "import params; print (params.rundir())"`
outdir=`python -c "import params; print (params.inputdir())"`
method=`python -c "import params; print (params.method())"`
beta=`python -c "import params; print (params.beta())"`
E=`python -c "import params; print (params.E())"`
alpha=`python -c "import params; print (params.alpha())"`
distopen=`python -c "import params; print (params.distopen())"`
diststd=`python -c "import params; print (params.diststd())"`
cpunums=`python -c "import params; print (params.cpu_nums())"`
CAMADIR=`python -c "import params; print (params.CaMa_dir())"`
#=================================================

# OpenMP Thread number
export OMP_NUM_THREADS=$cpu_nums

echo $syear
echo $eyear
echo $ens_num
echo $runname
echo $rundir
echo $outdir
echo $method
echo $beta
echo $E 
echo $alpha 
echo $distopen 
echo $diststd 
echo $cpunums
echo $CAMADIR

python ./src/prep_runoff.py $syear $eyear $ens_num $runname $rundir $outdir $method $beta $E $alpha $distopen $diststd $NCPUS $CAMADIR

# link the folder to ./CaMa_in
# need only of runoff was saved in another folder
#ln -sf $outdir/$runname ./CaMa_in/$runname 

#wait

#conda deactivate