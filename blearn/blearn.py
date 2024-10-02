#!/usr/bin/env python3

""" Main script to run blearn, generating a learning dataset in Blender"""


__author__ = "Marco Job"

import sys
import os
import bpy

# Setup the path to find modules
dn = os.path.dirname(__file__)

if dn not in sys.path:
    sys.path.append(dn)

# Import own code now
from src.blender import Blender  # NOQA


def main():
    b = Blender(config_file="config/config_road_corridor_demo.yml")
    b.start()


if __name__ == "__main__":
    main()
