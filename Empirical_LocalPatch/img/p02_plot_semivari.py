#!/opt/local/bin/python
# -*- coding: utf-8 -*-
""" make plots for fitted semivariograms
for each upstream and downstream
Menaka@IIS 2023/04/06"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib as mpl
import sys
import os
import matplotlib.gridspec as gridspec
import string
import calendar
import errno
import re
import math
from numpy import ma 
import matplotlib.gridspec as gridspec
#from mpl_toolkits.basemap import Basemap
#from slacker import Slacker
from multiprocessing import Pool
from multiprocessing import Process
#---
import LMA_semivari as LMA
#---


fname="../semivar/amz_06min_ERA5/00700086/up00001.svg" # Change File Here
with open(fname, "r") as f:
    lines = f.readlines()
ldis=[]
lgamma=[]
for line in lines[1::]:
    print (line)
    line = list(filter(None, re.split(" ",line)))
    dis  = float(line[2])
    gamma= float(line[3])
    ldis.append(dis)
    lgamma.append(gamma)
# plot
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(ldis,lgamma,marker="o",linestyle="None",markeredgecolor="k",markerfacecolor='none',markersize=5)


ax.set_xlabel('Distance')
ax.set_ylabel('Semi-Variance')
plt.show()

fig.savefig('p02_semi-variance.png',dpi=300)
