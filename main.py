import os
import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scripts import *

# Variableset
########################################################################################################################
imagedir=r''
labeldir=r''
resdir=r''
txtdir=r'' # init txt dir
imgsize=(1024,1024,3) # img size
lablltype='png' # jpg or png the output label type
roaddir=r'' # folder to save road label
nulldir=r'' # folder to save null label
roadsuffix='' # set the road result name's suffix
nullsuffix='' # set the null result name's suffix
roadwidth=10 # dilation size
########################################################################################################################

if __name__ == '__main__':
    createlineimage(transstrrecordtolist(getTruelabelfromTXT(txtdir)))
