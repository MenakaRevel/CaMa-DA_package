#===========================
#*** PBS setting when needed
#PBS -q E40
#PBS -l select=1:ncpus=40:mem=60gb
#PBS -j oe
#PBS -m ea
#PBS -M abdul.moiz@rainbow.iis.u-tokyo.ac.jp
#PBS -V
#PBS -N Altimap_test
#===========================
cd $PBS_O_WORKDIR
