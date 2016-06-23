import distutils
from distutils.core import setup

# The main call
setup(name='drawDECam',
      version ='0.2.0',
      license = "GPL",
      description = "A simple set of matplotlib API to draw DECam shapes using Plot and/or Polygons",
      author = "Felipe Menanteau",
      author_email = "felipe@illinois.edu",
      packages = ['drawDECam'],
      package_dir = {'': 'python'},
      )

