#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
from drawDECam import drawDECam as dDECam
import math

def drawGrid_DECam():

    # The center of the grid
    ra0  =  21.32981300
    dec0 = -44.56367600

    # Get the pattern
    ra,dec = dDECam.make_DECam_grid(ra0,dec0,nx=3,ny=3)

    d1 = dec.min() - 1.5
    d2 = dec.max() + 1.5
    d0 = (d2+d1)/2.0
    r1 = ra.min() - 1.5/math.cos(d0*math.pi/180.)
    r2 = ra.max() + 1.5/math.cos(d0*math.pi/180.)

    dec_range = abs(d2 - d1)
    ra_range  = abs(r2 - r1)

    print "# Fig 5: DECam Grid corners in the sky"
    aspect = dec_range/(ra_range*math.cos(d0*math.pi/180.))
    plt.figure(5,figsize=(10,10*aspect))

    for r,d in zip(ra,dec):
        dDECam.drawDECamCorners_Sky(r,d,pixscale=0.27,color='k',lw=0.5,ls='-')

    # Make the center in red
    dDECam.drawDECamCorners_Sky(ra0,dec0,pixscale=0.27,color='r',lw=1.5,ls='-')

    plt.xlim(r1,r2)
    plt.ylim(d1,d2)
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    plt.title("DECam Corners Sky")
    plt.show()

def examples_DECam():

    """
    A bunch of tests to make sure everything works with drawDECam
    """

    # These are the predefined centers of DECam
    x0 = dDECam.CCDSECTION_X0
    y0 = dDECam.CCDSECTION_Y0

    print "# Fig 1: DECam corners and CCDs using Polygons "
    plt.figure(1,figsize=(12,12))
    dDECam.drawDECamCorners_Polygon(x0,y0,rotate=True,color='red',closed=True,fill=False,ec='b',lw=0.5)
    dDECam.drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=False,ls='-',lw=0.5,color='k')
    dDECam.drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=True,ls='-', lw=0.5,color='r',alpha=0.1)
    plt.xlim(-15000,15000)
    plt.ylim(-15000,15000)
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.title("Rotated DECam")

    print "# Fig 2: DECam corners and CCDS using Plot "
    plt.figure(2,figsize=(12,12))
    dDECam.drawDECamCCDs_Plot(x0,y0,rotate=False,label=True, color='k',lw=0.5,ls='-')
    dDECam.drawDECamCorners_Plot(x0,y0,rotate=False,color='blue',lw=0.5,ls='-')
    plt.xlim(-15000,15000)
    plt.ylim(-15000,15000)
    plt.title("UN-Rotated DECam")
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')

    ra  = numpy.arange(10, 30,2.5)
    dec = numpy.arange(-3, 3,2.5)
    r1 = ra.min() - 1.5
    r2 = ra.max() + 1.5
    d1 = dec.min() - 1.5
    d2 = dec.max() + 1.5

    ra_range  = r2 - r1
    dec_range = d2 - d1
    aspect = dec_range/ra_range

    print "# Fig 3: DECam corners in the sky"
    plt.figure(3,figsize=(12,12*aspect))
    for r in ra:
        for d in dec:
            dDECam.drawDECamCorners_Sky(r,d,pixscale=0.27,label=False,color='k',lw=0.5,ls='-')
    plt.xlim(r1,r2)
    plt.ylim(d1,d2)
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    plt.title("DECam Corners Sky")

    print "# Fig 4: DECam CCDs in the sky"
    plt.figure(4,figsize=(12,12*aspect))
    for r in ra:
        for d in dec:
            dDECam.drawDECamCCDs_Sky(r,d,trim=True,pixscale=0.27,label=False,color='k',ls='-')
    plt.xlim(r1,r2)
    plt.ylim(d1,d2)
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    plt.title("DECam CCDs Sky")
    
    return

if __name__ == "__main__":
    examples_DECam()
    drawGrid_DECam()
    plt.show()
    
