#!/usr/bin/env python

"""
Purpose:
 A simple set of matplotlib API to draw DECam shapes using Plot and/or Polygons.

Requirements:
 matplotlib (obviously)
 numpy 
 despyastro (needs wcsutil only)
 
Examples:

  import drawDECam as dDECam
  import matplotlib.pyplot as plt

  # The Center of the Focal Plane
  x0 = dDECam.CCDSECTION_X0
  y0 = dDECam.CCDSECTION_Y0

  plt.figure(1,figsize=(9,9))

  # Rotated DECam
  # Draw DECam corners using Polygons at center
  dDECam.drawDECamCorners_Polygon(x0,y0,rotate=True,color='red',closed=True,fill=False,ec='b',lw=0.5)

  # Draw DECam CCDS using Polygons no filling
  dDECam.drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=False,ls='-',lw=0.5,color='k')

  # Draw DECam CCDS using Polygons fill
  dDECam.drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=True,ls='-', lw=0.5,color='r',alpha=0.1)

  # Un-Rotated DECam
  # Draw DECam CCDs using Plot function (Unrotated)
  drawDECamCCDs_Plot(x0,y0,rotate=False,label=True, color='k',lw=0.5,ls='-')

  # Draw DECam Corners using Plot function (Unrotated)
  drawDECamCorners_Plot(x0,y0,rotate=False,color='blue',lw=0.5,ls='-')

  # DECam Corners on sky -- uses simple 'fake' RA-TAN projection
  drawDECamCorners_Sky(ra,dec,pixscale=0.27,label=False,color='k',lw=0.5,ls='-')

  # DECam CCDs
  drawDECamCCDs_Sky(ra,dec,trim=True,pixscale=0.27,label=False,color='k',ls='-')


Author:
 Felipe Menanteau, NCSA, April 2014

"""

import math
# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.patches     import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
# wcsutil to go to the sky!
import numpy
from despyastro import wcsutil 

################################################ Hard Coded constants
### for DECam, a set of list/arrays and dictionaries are hard-coded
### here
#
#
# CCDSECTIONS:
# A dictionary of all of the CCDSECTIONS from a DECam image, produced like:
# This is how I got the CCDSECTIONS for an exposure:
#  gethead -up DETSEC CCDNUM CRPIX1 CRPIX2 CRVAL1 CRVAL2 PIXSCAL1 DE*[0-9].fits
# | awk '{printf \"%s    %-30s %2d    %12.2f %12.2f %14.8f %14.8f\n\",$1,$2,$3,$4,$5,$6,$7}'
#

CCDSECTIONS = {
    1  : [2049,4096,8193,12288],
    2  : [2049,4096,12289,16384],
    3  : [2049,4096,16385,20480],
    4  : [4097,6144,6145,10240],
    5  : [4097,6144,10241,14336],
    6  : [4097,6144,14337,18432],
    7  : [4097,6144,18433,22528],
    8  : [6145,8192,4097,8192],
    9  : [6145,8192,8193,12288],
    10 : [6145,8192,12289,16384],
    11 : [6145,8192,16385,20480],
    12 : [6145,8192,20481,24576],
    13 : [8193,10240,2049,6144],
    14 : [8193,10240,6145,10240],
    15 : [8193,10240,10241,14336],
    16 : [8193,10240,14337,18432],
    17 : [8193,10240,18433,22528],
    18 : [8193,10240,22529,26624],
    19 : [10241,12288,2049,6144],
    20 : [10241,12288,6145,10240],
    21 : [10241,12288,10241,14336],
    22 : [10241,12288,14337,18432],
    23 : [10241,12288,18433,22528],
    24 : [10241,12288,22529,26624],
    25 : [12289,14336,1,4096],
    26 : [12289,14336,4097,8192],
    27 : [12289,14336,8193,12288],
    28 : [12289,14336,12289,16384],
    29 : [12289,14336,16385,20480],
    30 : [12289,14336,20481,24576],
    31 : [12289,14336,24577,28672],
    32 : [14337,16384,1,4096],
    33 : [14337,16384,4097,8192],
    34 : [14337,16384,8193,12288],
    35 : [14337,16384,12289,16384],
    36 : [14337,16384,16385,20480],
    37 : [14337,16384,20481,24576],
    38 : [14337,16384,24577,28672],
    39 : [16385,18432,2049,6144],
    40 : [16385,18432,6145,10240],
    41 : [16385,18432,10240,14335],
    42 : [16385,18432,14336,18431],
    43 : [16385,18432,18432,22527],
    44 : [16385,18432,22528,26623],
    45 : [18433,20480,2049,6144],
    46 : [18433,20480,6145,10240],
    47 : [18433,20480,10240,14335],
    48 : [18433,20480,14336,18431],
    49 : [18433,20480,18432,22527],
    50 : [18433,20480,22528,26623],
    51 : [20481,22528,4097,8192],
    52 : [20481,22528,8193,12288],
    53 : [20481,22528,12289,16384],
    54 : [20481,22528,16385,20480],
    55 : [20481,22528,20481,24576],
    56 : [22529,24576,6145,10240],
    57 : [22529,24576,10240,14335],
    58 : [22529,24576,14336,18431],
    59 : [22529,24576,18432,22527],
    60 : [24577,26624,8193,12288],
    61 : [24577,26624,12289,16384],
    62 : [24577,26624,16385,20480]
    }

CCDSECTION_X0 = (CCDSECTIONS[28][1]+CCDSECTIONS[35][0])/2.0
CCDSECTION_Y0 = (CCDSECTIONS[35][2]+CCDSECTIONS[28][3])/2.0

# Create the TRIM_CCDSECTION
TRIM_CCDSECTIONS = CCDSECTIONS.copy()
borderpix = 104 # 208/2. as 208 is the space between chips in pixels
for _k,_v in TRIM_CCDSECTIONS.items():
    (_x1,_x2,_y1,_y2) = _v
    _x1 = _x1 + borderpix
    _x2 = _x2 - borderpix
    _y1 = _y1 + borderpix
    _y2 = _y2 - borderpix
    TRIM_CCDSECTIONS[_k] = [_x1,_x2,_y1,_y2]

# Create the Corners of DECam Footprint
DECAM_CORNERS_X = []
DECAM_CORNERS_Y = []
# Bottom [L/R]
for _k in [1,4,8,13,19,25,32,39,45,51,56,60]:
    DECAM_CORNERS_X.append(CCDSECTIONS[_k][0])
    DECAM_CORNERS_X.append(CCDSECTIONS[_k][1])
    DECAM_CORNERS_Y.append(CCDSECTIONS[_k][2])
    DECAM_CORNERS_Y.append(CCDSECTIONS[_k][2])
# Top [L/R]
for _k in [62,59,55,50,44,38,31,24,18,12,7,3]:
    DECAM_CORNERS_X.append(CCDSECTIONS[_k][1])
    DECAM_CORNERS_X.append(CCDSECTIONS[_k][0])
    DECAM_CORNERS_Y.append(CCDSECTIONS[_k][3])
    DECAM_CORNERS_Y.append(CCDSECTIONS[_k][3])
# close the points
DECAM_CORNERS_X.append(DECAM_CORNERS_X[0])
DECAM_CORNERS_Y.append(DECAM_CORNERS_Y[0])
# Into numpy arrays
DECAM_CORNERS_X = numpy.array(DECAM_CORNERS_X)
DECAM_CORNERS_Y = numpy.array(DECAM_CORNERS_Y)

######  End of hard-coded constants for DECam #######

def DECamFootprint(CCDSECTIONS):

    """ Produces DECAM Layout -- not used, but needed to create DECAM_CORNERS"""

    xb = []
    yb = []
    # Bottom [L/R]
    for k in [1,4,8,13,19,25,32,39,45,51,56,60]:
        xb.append(CCDSECTIONS[k][0])
        xb.append(CCDSECTIONS[k][1])
        yb.append(CCDSECTIONS[k][2])
        yb.append(CCDSECTIONS[k][2])
    # Top [L/R]
    for k in [62,59,55,50,44,38,31,24,18,12,7,3]:
        xb.append(CCDSECTIONS[k][1])
        xb.append(CCDSECTIONS[k][0])
        yb.append(CCDSECTIONS[k][3])
        yb.append(CCDSECTIONS[k][3])
    # close the points
    xb.append(xb[0])
    yb.append(yb[0])
    return xb,yb

def drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=False,**kwargs):

    """ Draws DECam CCDs shapes using matplotlib Polygons function on the current plot"""

    ax = plt.gca()
    if trim:
        SECTIONS = TRIM_CCDSECTIONS
    else:
        SECTIONS = CCDSECTIONS

    patches = []
    for k,v in SECTIONS.items():
        (x1,x2,y1,y2) = v
        if rotate:
            x1,y1 = rotate_xy(x1,y1,theta=-90,x0=x0,y0=y0)
            x2,y2 = rotate_xy(x2,y2,theta=-90,x0=x0,y0=y0)
        else:
            x1,y1 = rotate_xy(x1,y1,theta=0,x0=x0,y0=y0)
            x2,y2 = rotate_xy(x2,y2,theta=0,x0=x0,y0=y0)

        x = numpy.array([x1,x2,x2,x1])
        y = numpy.array([y1,y1,y2,y2])
        P = Polygon(zip(x,y),**kwargs)
        patches.append(P)  
        if label:
            ax.text(0.5*(x2+x1),0.5*(y2+y1),"CCD%s" % k, ha='center',va='center')

    p = PatchCollection(patches,match_original=True)
    ax.add_collection(p)
    return


def drawDECamCCDs_Plot(x0,y0,trim=True,rotate=True,label=False,**kwargs):

    """ Draws DECam CCDs shapes using matplotlib Plot function on the current plot"""

    ax = plt.gca()
    if trim:
        SECTIONS = TRIM_CCDSECTIONS
    else:
        SECTIONS = CCDSECTIONS
    for k,v in SECTIONS.items():
        (x1,x2,y1,y2) = v
        if rotate:
            x1,y1 = rotate_xy(x1,y1,theta=-90,x0=x0,y0=y0)
            x2,y2 = rotate_xy(x2,y2,theta=-90,x0=x0,y0=y0)
        else:
            x1,y1 = rotate_xy(x1,y1,theta=0,x0=x0,y0=y0)
            x2,y2 = rotate_xy(x2,y2,theta=0,x0=x0,y0=y0)

        # Into numpy arrays
        x = numpy.array([x1,x2,x2,x1,x1])
        y = numpy.array([y1,y1,y2,y2,y1])
        ax.plot(x,y,**kwargs)
        if label:
            ax.text(0.5*(x2+x1),0.5*(y2+y1),"CCD%s" % k, ha='center',va='center')
    return


def drawDECamCorners_Plot(x0,y0,rotate=True,**kwargs):

    """ Draws DECam Corners using matplotlib Plot function on the current plot"""

    ax = plt.gca()
    if rotate:
        x,y = rotate_xy(DECAM_CORNERS_X,DECAM_CORNERS_Y,theta=-90,x0=x0,y0=y0)
    else:
        x,y = rotate_xy(DECAM_CORNERS_X,DECAM_CORNERS_Y,theta=  0,x0=x0,y0=y0)
    ax.plot(x,y,**kwargs)
    return

def drawDECamCorners_Polygon(x0,y0,rotate=True,draw=True,**kwargs):

    """ Draws DECam Corners using matplotlib Plot function on the current plot"""
    ax = plt.gca()
    if rotate:
        x,y = rotate_xy(DECAM_CORNERS_X,DECAM_CORNERS_Y,theta=-90,x0=x0,y0=y0)
    else:
        x,y = rotate_xy(DECAM_CORNERS_X,DECAM_CORNERS_Y,theta=  0,x0=x0,y0=y0)
    P = Polygon(zip(x[:-1],y[:-1]),**kwargs)
    ax.add_patch(P)
    if draw: plt.draw()
    return

def drawDECamCorners_Sky(ra,dec,pixscale=0.27,label=False,**kwargs):

    """ Draws DECam Corners on the sky (RA,DEC) using matplotlib Plot function on the current plot"""

    header = createDECam_TANheader(ra,dec,pixscale=pixscale)
    wcs    = wcsutil.WCS(header)

    r,d = wcs.image2sky(DECAM_CORNERS_X,DECAM_CORNERS_Y) # Into RA,DEC
    ax = plt.gca()
    ax.plot(r,d,**kwargs)
    if label: # Probably will never use
        r0,d0 =  wcs.image2sky(ra,dec)
        ax.text(r,d, label, ha='center',va='center')
    return

def drawDECamCCDs_Sky(ra,dec,trim=True,pixscale=0.27,label=False,**kwargs):

    """ Draws DECam CCDs on the sky (RA,DEC) using matplotlib Plot function on the current plot"""
    
    header = createDECam_TANheader(ra,dec,pixscale=pixscale)
    wcs = wcsutil.WCS(header)

    ax = plt.gca()
    if trim:
        SECTIONS = TRIM_CCDSECTIONS
    else:
        SECTIONS = CCDSECTIONS
    for k,v in SECTIONS.items():
        (x1,x2,y1,y2) = v
        x = numpy.array([x1,x2,x2,x1,x1])
        y = numpy.array([y1,y1,y2,y2,y1])
        r,d = wcs.image2sky(x,y) # Into RA,DEC
        ax.plot(r,d,**kwargs)
        if label: # Probably will never use
            r1,d1 =  wcs.image2sky(x1,y1)
            r2,d2 =  wcs.image2sky(x2,y2)
            ax.text(0.5*(r2+r1),0.5*(d2+d1),"CCD%s" % k, ha='center',va='center')
    return


def createDECam_TANheader(ra_center,dec_center,pixscale=0.27):
    """
    Creates a fake TAN projection header for DECam image to project the CCD Sections on the sky
    """
    DECam_header = {
        'CTYPE1'  : 'RA---TAN',       #/ WCS projection type for this axis
        'CTYPE2'  : 'DEC--TAN',       #/ WCS projection type for this axis
        'CUNIT1'  : 'deg',            #/ Axis unit
        'CUNIT2'  : 'deg',            #/ Axis unit
        'CRVAL1'  :  ra_center,       #/ World coordinate on this axis
        'CRPIX1'  :  CCDSECTION_X0,   #/ Reference pixel on this axis
        'CD1_1'   :  0,               #/ 
        'CD1_2'   :  +pixscale/3600., #/ 
        'CRVAL2'  :  dec_center,      #/ 
        'CRPIX2'  :  CCDSECTION_Y0,   #/ 
        'CD2_1'   :  -pixscale/3600., #/ 
        'CD2_2'   :  0.               #/ 
        }

    # Now for consistecy, add lower-case keys, to make it 'case-insensity' fake
    for k, v in DECam_header.items():
        DECam_header[k.lower()] = v
    return DECam_header


def getDECamCorners(header):

    """
    Get the DECam profile corners for a given header and WCS
    """

    wcs    = wcsutil.WCS(header)
    nx = header['NAXIS1']
    ny = header['NAXIS2']
    ra0,dec0 =  wcs.image2sky(nx/2.0,ny/2.0)
    DECam_header = createDECam_TANheader(ra0,dec0)
    DECam_wcs    = wcsutil.WCS(DECam_header)
    r,d = DECam_wcs.image2sky(DECAM_CORNERS_X,DECAM_CORNERS_Y)
    x,y = wcs.sky2image(r,d)
    return x,y

def getDECamCCDs(header,plot=False,trim=True,**kwargs):

    """
    Get the x,y positions of the DECam CCDs for a given header
    with WCS and plot them (optional)
    """

    wcs    = wcsutil.WCS(header)
    nx = header['NAXIS1']
    ny = header['NAXIS2']
    ra0,dec0 =  wcs.image2sky(nx/2.0,ny/2.0)
    DECam_header = createDECam_TANheader(ra0,dec0)
    DECam_wcs    = wcsutil.WCS(DECam_header)

    if plot:
        ax = plt.gca()
    if trim:
        SECTIONS = TRIM_CCDSECTIONS
    else:
        SECTIONS = CCDSECTIONS

    ccds = {}
    for k,v in SECTIONS.items():
        (x1,x2,y1,y2) = v
        if plot:
            DECam_x = [x1,x2,x2,x1,x1]
            DECam_y = [y1,y1,y2,y2,y1]
            r_plot,d_plot = DECam_wcs.image2sky(DECam_x,DECam_y) # Into RA,DEC
            x_plot,y_plot = wcs.sky2image(r_plot,d_plot)
            ax.plot(x_plot,y_plot,**kwargs)
            
        r,d = DECam_wcs.image2sky([x1,x2],[y1,y2])
        x,y = wcs.sky2image(r,d)
        # Make the the interger (pixels)
        x = numpy.ceil(x).astype(int)
        y = numpy.ceil(y).astype(int)
        # Sort them to get the slices right
        x.sort()
        y.sort()
        ccds[k] = x.tolist(),y.tolist() # We get back lists
    return ccds



def DECamMask(header,plot=False,**kw):

    """
    Make a mask with 1 or 0 with the shape of DECam for an input
    header object with a WCS
    """
    # Get the ccds
    ccds = getDECamCCDs(header,plot=plot,**kw)

    nx = header['NAXIS1']
    ny = header['NAXIS2']
    mask = numpy.zeros((ny,nx),dtype=int)
    for k in ccds.keys():
        # Unpack the edges
        [x1,x2],[y1,y2] = ccds[k]
        mask[y1:y2,x1:x2] = 1

    return mask


def DECamMask2(filename,plot=False,**kw):

    import fitsio

    data,header = fitsio.read(filename, ext=0, header=True)
    print "# Done reading %s" % filename

    newdata = (data*0).astype('float32')

    """
    Make a mask with 1 or 0 with the shape of DECam for an input
    header object with a WCS
    """
    # Get the ccds
    ccds = getDECamCCDs(header,plot=plot,**kw)

    #nx = header['NAXIS1']
    #ny = header['NAXIS2']
    #mask = numpy.zeros((ny,nx),dtype=int)
    for k in ccds.keys():
        # Unpack the edges
        [x1,x2],[y1,y2] = ccds[k]
        newdata[y2:y1,x2:x1] = data[y2:y1,x2:x1]  
    return newdata



def rotate_xy(x,y,theta,x0=0,y0=0,units='degrees'):
    """
    Rotates (x,y) by angle theta and (x0,y0) translation
    """
    if units == 'degrees':
        d2r = math.pi/180. # degrees to radians shorthand
        theta = d2r*theta
    x_new =  (x-x0)*math.cos(theta) - (y-y0)*math.sin(theta)
    y_new =  (x-x0)*math.sin(theta) + (y-y0)*math.cos(theta)
    return x_new,y_new

#################################################################
# Extra useful function to draw ellipses and circles as polygons
#################################################################

def PEllipse((xo,yo), (A, B), resolution=100, angle=0.0, **kwargs):

    """ Draws one ellipse at a time -- slower """

    pi    = math.pi
    cos   = math.cos
    sin   = math.sin
    angle = (angle+90)*math.pi/180. # for SDSS 90+theta
 
    t    = 2*pi/resolution*numpy.arange(resolution)
    xtmp = A*numpy.cos(t)
    ytmp = B*numpy.sin(t)

    x = xtmp*cos(angle) - ytmp*sin(angle) + xo
    y = xtmp*sin(angle) + ytmp*cos(angle) + yo
    return Polygon(zip(x, y), **kwargs)

def PEllipse_multi((xo,yo), (A, B), resolution=100, angle=0.0, **kwargs):

    """ Draw ellipse patches created all at once, uses numpy arrays """

    pi    = math.pi
    cos   = math.cos
    sin   = math.sin
    angle = (angle+90)*math.pi/180. # for SDSS 90+theta
    t     = 2*pi/resolution*numpy.arange(resolution)

    verts = []
    for k in range(len(xo)):
        xtmp = A[k]*numpy.cos(t)
        ytmp = B[k]*numpy.sin(t)
        x = xtmp*cos(angle[k]) - ytmp*sin(angle[k]) + xo[k]
        y = xtmp*sin(angle[k]) + ytmp*cos(angle[k]) + yo[k]
        verts.append(zip(x,y))

    verts = numpy.array(verts)
    coll = PolyCollection(verts,**kwargs)
    plt.gca().add_collection(coll)
    return 

def PCircle((xo,yo), radius, resolution=100, **kwargs):

    """ Draw a circle as a Polygon too """

    pi    = math.pi
    cos   = math.cos
    sin   = math.sin
    t    = 2*pi/resolution*numpy.arange(resolution)
    xtmp = radius*numpy.cos(t)
    ytmp = radius*numpy.sin(t)
    x = xtmp + xo;
    y = ytmp + yo;
    return Polygon(zip(x, y), **kwargs)

def run_tests():

    """
    Performs a bunch of tests to make sure everything works
    """

    # These are the predefined centers of DECam
    x0 = CCDSECTION_X0
    y0 = CCDSECTION_Y0

    print "# Fig 1: DECam corners and CCDs using Polygons "
    plt.figure(1,figsize=(12,12))
    drawDECamCorners_Polygon(x0,y0,rotate=True,color='red',closed=True,fill=False,ec='b',lw=0.5)
    drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=False,ls='-',lw=0.5,color='k')
    drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=True,ls='-', lw=0.5,color='r',alpha=0.1)
    plt.xlim(-15000,15000)
    plt.ylim(-15000,15000)
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.title("Rotated DECam")

    print "# Fig 2: DECam corners and CCDS using Plot "
    plt.figure(2,figsize=(12,12))
    drawDECamCCDs_Plot(x0,y0,rotate=False,label=True, color='k',lw=0.5,ls='-')
    drawDECamCorners_Plot(x0,y0,rotate=False,color='blue',lw=0.5,ls='-')
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
            drawDECamCorners_Sky(r,d,pixscale=0.27,label=False,color='k',lw=0.5,ls='-')
    plt.xlim(r1,r2)
    plt.ylim(d1,d2)
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    plt.title("DECam Corners Sky")

    print "# Fig 4: DECam CCDs in the sky"
    plt.figure(4,figsize=(12,12*aspect))
    for r in ra:
        for d in dec:
            drawDECamCCDs_Sky(r,d,trim=True,pixscale=0.27,label=False,color='k',ls='-')
    plt.xlim(r1,r2)
    plt.ylim(d1,d2)
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    plt.title("DECam CCDs Sky")
    
    plt.show()
    return


if __name__ == "__main__":
    run_tests()
