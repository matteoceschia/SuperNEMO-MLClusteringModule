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

## Prerequisites

A working version of Falaise, and the installation of the header-only library frugally-deep.

## Description

Add to an flreconstruct pipeline to cluster tracker hits for reconstruction.

First of all, you need to edit lines 24-25 the CMakeLists.txt file. Do

``` console
$ python-config --cflags
```
and copy the path after -I in line 24 (i.e. edit include_directories(...)). Then, do

``` console
$ python-config --ldflags
```
and copy the path after -L in line 25 (i.e. edit LINK_DIRECTORIES(...)).

To build it, do

``` console
$ mkdir build
$ cd build
$ cmake ..
...
$ make
...
```

## Usage

To run it, for the moment you MUST be in the home directory of the project, where py_clustering_module.py is found (i.e. NOT in the build). Then do

``` console
$ flreconstruct -i input.brio -p mlcModuleExample.conf -o output.brio
```
