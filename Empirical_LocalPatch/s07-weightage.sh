#!/bin/sh
#====================
# Calcualte the experimental semivarinces along each river stem
# Menaka@IIS
# 2020/06/01
#====================
#*** PBS setting when needed
#PBS -q F40
#PBS -l select=1:ncpus=40:mem=60gb
#PBS -j oe
#PBS -m ea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N weight
#========
#cd $PBS_O_WORKDIR
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
inputname=`python -c "import params; print (params.input_name())"`
N=`python src/calc_days.py $syear $smonth $sdate $eyear $emonth $edate`
threshold=`python -c "import params; print (params.threshold())"`
# represnt dams
damrep=`python -c "import params; print (params.dam_rep())"`
#=================================================

# Get Dam List for Regionalized CaMa-Flood
if [ $damrep == "1" ]
then
    cd ./etc/
    ./alloc_dam.sh $CAMADIR "glb_06min"
    ./regionalize_dam.sh $CAMADIR "glb_06min" $mapname
    cd ../
fi

python src/weightage.py $CAMADIR $mapname $inputname $outdir $cpunums $threshold $damrep
