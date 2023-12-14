#!/bin/sh
#====================
# create statitcs (e.g., mean, std) from netCDF4 files
# dimesnion (nx, ny)
# Menaka@IIS
# 2020/05/29
#===========================
#*** PBS setting when needed
#PBS -q E20
#PBS -l select=1:ncpus=20:mem=186gb
#PBS -j oe
#PBS -m ea
#PBS -M menaka@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N mk_stat
#===========================
# cd $PBS_O_WORKDIR
#================================================
source ~/.bashrc
conda deactivate
conda activate
which python

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
CAMADIR=`python -c "import params; print (params.CaMa_dir())"`
outdir="./" #`python -c "import params; print (params.out_dir())"`
cpunums=`python -c "import params; print (params.cpu_nums())"`
mapname=`python -c "import params; print (params.mapname())"`
expname=`python -c "import params; print (params.expname())"`
runname=`python -c "import params; print (params.runname())"`
ens_num=`python -c "import params; print (params.ens_mem())"`
#=================================================
# OpenMP Thread number
export OMP_NUM_THREADS=$cpunums
# run make statistics
echo "python ./src/stat_sfcelv.py $ssyear $eeyear $expname $runname $mapname $ens_num $NCPUS $ssyear $eeyear $outdir"
python ./src/stat_sfcelv.py $ssyear $eeyear $expname $runname $mapname $ens_num $NCPUS $ssyear $eeyear $outdir

#=================================================
wait

conda deactivate