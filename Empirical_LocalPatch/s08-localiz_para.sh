#!/bin/sh
#====================
# Write localization parameters to easy acess text files
# Menaka@IIS
# 2020/06/01
#====================
#*** PBS setting when needed
#PBS -q F10
#PBS -l select=1:ncpus=10:mem=40gb
#PBS -j oe
#PBS -m ea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N lpara
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
echo $syear" to "$eyear
CAMADIR=`python -c "import params; print (params.CaMa_dir())"`
outdir=`python -c "import params; print (params.out_dir())"`
cpunums=`python -c "import params; print (params.cpu_nums())"`
mapname=`python -c "import params; print (params.map_name())"`
# represnt dams
damrep=`python -c "import params; print (params.dam_rep())"`
inputname=`python -c "import params; print (params.input_name())"`
N=`python src/calc_days.py $syear $smonth $sdate $eyear $emonth $edate`
threshold=`python -c "import params; print (params.threshold())"`
patch=`python -c "import params; print (params.patch())"`

threshname=$(echo $threshold 100 | awk '{printf "%2d\n",$1*$2}')

# OpenMP Thread number
export OMP_NUM_THREADS=$cpunums

# make dir local patch
if [ ${damrep} -eq 1 ]; then
    mkdir -p "./local_patch/${mapname}_${inputname}_${threshname}_dam"
else
    mkdir -p "./local_patch/${mapname}_${inputname}_${threshname}"
fi

#=================================================
# Write local patch parameters
echo "./src/lpara $N $syear $eyear $mapname $inputname $CAMADIR $outdir $threshold $patch ${damrep} $cpunums"
./src/lpara $N $syear $eyear $mapname $inputname $CAMADIR $outdir $threshold $patch ${damrep} $cpunums
