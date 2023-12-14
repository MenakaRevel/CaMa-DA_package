#! /usr/bin/bash
########################
#
# this program run the whole program
#
########################
#*** PBS seeting when needed
#PBS -q F20
#PBS -l select=1:ncpus=20:mem=40gb
#PBS -j oe
#PBS -m bea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N Ensemble_Sim
# cd $PBS_O_WORKDIR

orgdir=`python -c "import params; print (params.org_dir())"`
cpunums=`python -c "import params; print (params.cpu_nums())"`
echo ${orgdir}

# # copy source files
# cp -r ${orgdir}/params.py        ./params.py 
# cp -r ${orgdir}/run.py           ./run.py
# cp -r ${orgdir}/main_code.py     ./main_code.py
# cp -r ${orgdir}/prep_runoff.py   ./prep_runoff.py

# link source codes
rm -rf ./run.py
rm -rf ./main_code.py

ln -sf ${orgdir}/src/run.py          ./run.py
ln -sf ${orgdir}/src/main_code.py    ./main_code.py
ln -s ${orgdir}/params.py ${orgdir}/src/params.py

# run the simulations
echo ${orgdir}
echo "running the simulations..."

export OMP_NUM_THREADS=$cpunums
python run.py 

wait

echo "simulations done."
