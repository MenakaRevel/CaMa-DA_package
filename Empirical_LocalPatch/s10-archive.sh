#!/bin/sh
#====================
# Archive intermediate files
# Moiz@IIS
# 2023/07/26
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


if [[ $damrep == 1 ]]
then
    arch_dir=${mapname}_${inputname}_${threshname}_dam
else
    arch_dir=${mapname}_${inputname}_${threshname}
fi

echo Compressing Files...
tar -zcvf  CaMa_out/${mapname}_${inputname}.tar.gz ./CaMa_out/${mapname}_${inputname}
tar -zcvf  ./semivar/${mapname}_${inputname}.tar.gz ./semivar/${mapname}_${inputname}
tar -zcvf  ./weightage/${arch_dir}.tar.gz ./weightage/${arch_dir}
tar -zcvf  ./gaussian_weightage/${arch_dir}.tar.gz ./gaussian_weightage/${arch_dir}

