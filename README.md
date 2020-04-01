# MLClustering Module readme


Matteo Ceschia (UCL)
Last updated March 31, 2020

The MLClustering module is a reconstruction module. It attempts to
cluster tracker hits and fill the TCD data bank in Falaise in the current
event data model.


## Files:

- mlclustering_library.cpp
- mlclustering_library.h
- mlclustering_module.cpp
- mlclustering_module.h
- CMakeLists.txt
- mlcModuleExample.conf
- fdeep_model.json (CNN model architecture)

##Prerequisites

A working version of Falaise, and the installation of the header-only library frugally-deep.

## Description

Add to an flreconstruct pipeline to cluster tracker hits for reconstruction. To build it, do

``` console
$ mkdir build
$ cd build
$ cmake ..
...
$ make
...
```
