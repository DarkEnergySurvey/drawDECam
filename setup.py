import distutils
from distutils.core import setup

# The main call
setup(name='drawDECam',
      version ='0.1.0',
      license = "GPL",
      description = "A simple set of matplotlib API to draw DECam shapes using Plot and/or Polygons",
      author = "Felipe Menanteau",
      author_email = "felipe@illinois.edu",
      packages = ['despyastro'],
      package_dir = {'': 'python'},
      )

