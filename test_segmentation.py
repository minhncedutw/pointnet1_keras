'''
    File name: Test Segmentation
    Author: minhnc
    Date created(MM/DD/YYYY): 10/2/2018
    Last modified(MM/DD/YYYY HH:MM): 10/2/2018 5:22 AM
    Python Version: 3.6
    Other modules: [None]

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"


#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        for i in range(len(argv) - 1):
            print(argv[i + 1])


if __name__ == '__main__':
    main()
