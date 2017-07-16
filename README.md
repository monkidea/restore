# restore.py

Python 3 scripts for the automatic restoration of scans of age-deteriorated colour slides / prints.

This is an authorized GitHub mirror of [original code](http://www.lionhouse.plus.com/photosoftware/restore/) written by Geoff Daniell. The aim is make it easier for others to improve and extend this code, so pull requests are welcomed!

Note that only the [standalone Python 3](http://www.lionhouse.plus.com/photosoftware/restore/stand_alone/python3/) version can be found here. The Gimp plugins and Python 2 versions may be added at a future time.

# Getting Started

**Note this version is a WIP and may not function at all! The original code is likely to be more usable.**

# Requirements

* Python 3
* PIL

# Usage

`python .\main.py dir=bad_images`

The above command will process all the images in the directory 'bad_images' and place the restored images in 'bad_images/restored' (note 'restored' directory must exist).

To use the custom colour quantisation algorithm (vs. the one in PIL), change the `if 1` statement in restore.py to `if 0`.

For the theory behind these algorithms, please see [here](http://www.lionhouse.plus.com/photosoftware/restore/documents/).

# To-do

1. Improve command-line interface
1. General code cleanup
1. Replace Python algorithms with standard Python implementations (from NumPy, SciPy, etc.) for speed
1. Split into smaller modules that can be shared between different versions

# Credit

All credit to the original author Geoff Daniell.

