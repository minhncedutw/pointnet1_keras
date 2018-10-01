'''
    File name: Mayavi 3D points visualization
    Author: minhnc
    Date created(MM/DD/YYYY): 9/28/2018
    Last modified(MM/DD/YYYY HH:MM): 9/28/2018 5:18 AM
    Python Version: 3.5
    Other modules: []
    Reference: [Guide](https://www.scipy-lectures.org/advanced/3d_plotting/index.html#id4)

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import numpy as np
from mayavi import mlab

#==============================================================================
# Constant Definitions
#==============================================================================
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_PINK = (255, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
#==============================================================================
# Function Definitions
#==============================================================================
def standard_color(transparent=255):
    colors = np.array(
        [COLOR_BLACK, COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_PINK, COLOR_CYAN, COLOR_WHITE])
    colors = np.c_[colors, np.ones((len(colors))) * transparent]  # Add transparent value
    # print(colors)
    return colors


def random_color(length, transparent=255):
    # Define color table (including alpha), which must be uint8 and [0,255]
    colors = (np.random.random((length, 4)) * transparent).astype(np.uint8)
    colors[:, -1] = 255  # No transparency
    return colors


def visualize(x, y, z, label):
    N = len(x)

    ones = np.ones(N)
    pts = mlab.quiver3d(x, y, z, ones, ones, ones, scalars=label, mode='sphere')  # Create points
    pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar

    # colors = random_color(length=N)
    colors = standard_color()

    # Set look-up table and redraw
    pts.module_manager.scalar_lut_manager.lut.table = colors
    mlab.show()

    return 0

def custom_visulatilation():
    # Primitives
    N = 100  # Number of points

    # Create point cloud
    x= np.random.random((N))
    y= np.random.random((N))
    z= np.random.random((N))

    # Create label
    label = np.random.randint(low=0, high=8, size=(N))

    visualize(x=x, y=y, z=z, label=label)

    return 0
#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is Mayavi 3D points visualization Program')

    custom_visulatilation()


if __name__ == '__main__':
    main()
