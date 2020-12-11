# drawDECam

A simple set of matplotlib API to draw DECam shapes using Plot and/or Polygons.

## Requirements:
 - matplotlib (obviously)
 - numpy
 - despyastro (needs wcsutil only)

## Examples:

  ```
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
  dDECam.drawDECamCCDs_Polygon(x0,y0,trim=True,rotate=True,label=True,fill=True,
  ls='-', lw=0.5,color='r',alpha=0.1)

  # Un-Rotated DECam
  # Draw DECam CCDs using Plot function (Unrotated)
  drawDECamCCDs_Plot(x0,y0,rotate=False,label=True, color='k',lw=0.5,ls='-')

  # Draw DECam Corners using Plot function (Unrotated)
  drawDECamCorners_Plot(x0,y0,rotate=False,color='blue',lw=0.5,ls='-')

  # DECam Corners on sky -- uses simple 'fake' RA-TAN projection
  drawDECamCorners_Sky(ra,dec,pixscale=0.27,label=False,color='k',lw=0.5,ls='-')

  # DECam CCDs
  drawDECamCCDs_Sky(ra,dec,trim=True,pixscale=0.27,label=False,color='k',ls='-')```
