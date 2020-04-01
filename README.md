# MLClustering Module readme


Matteo Ceschia (UCL)
Last updated March 31, 2020

The MLClustering module is a reconstruction module. It attempts to
cluster tracker hits and fill the TCD data bank in Falaise in the current
event data model.


## Files:

- ImageSegmentation_library.cpp
- ImageSegmentation_library.h
- ImageSegmentation_clustering_module.cpp
- ImageSegmentation_clustering_module.h
- CMakeLists.txt
- iseg.conf


## Description

Add to an flreconstruct pipeline to cluster tracker hits for reconstruction. To build it, do

``` console
$ ls
CMakeLists.txt                   ImageSegmentation_library.h
README.md                        ImageSegmentation_clustering_module.cpp
ImageSegmentation_library.cpp	 ImageSegmentation_clustering_module.h
iseg.conf
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=$(brew --prefix) ..
...
$ make
...
... If you are developing the module, you can test it by calling
$ make test
...
... or obtain more detail on the tests and launch in the build directory
$ ctest -V
```

Note: if you get a QT5 error, you may need to specify the QT5 path when you run the cmake line, as given by `brew --prefix qt5-base`. For example, you can run:
``` console
$ cmake -DCMAKE_PREFIX_PATH="$(brew --prefix qt5-base);$(brew --prefix)" ..
```

The build will create the `libImageSegmentation.so` shared library. Assuming that you have an `input.brio` file that contains
a `CD` bank from the simulation or a data run, this can be run after editing
the configuration file to point at the library location. Note that the module
requires no parameters.:

``` console
...
[name="flreconstruct.plugins" type="flreconstruct::section"]
plugins : string[1] = "ImageSegmentation"
ImageSegmentation.directory : string = "/my/path/to/build/directory"

# Define the modules in the pipeline:
[name="pipeline" type="imagesegmentation_clustering_module"]
...
$ flreconstruct -i /path/to/input.brio -p iseg.conf -o /tmp/clustered_data.brio
```

## Implementation

This reconstruction module attempts to use the bare required minimum to run in
the Falaise pipeline and use the existing data model. None of the pre-defined
'base' classes are used other than the dpp::base_module which renders this
code into a Falaise module.

Input is the calibrated data bank, default name 'CD'. Output is put into the
default 'TCD' data bank, the tracker cluster data bank. The
TrackerPreClustering is not used since the identical functionality (setting
tracker hits as prompt or delayed, configurably with the delay time threshold)
is already implemented by the digitization hence hits can simply be split at the
start into prompt or delayed as they are read in from the 'CD' data bank.

This implementation uses the module process() method to actually do work and then
call the algorithm object itself rather than bringing in a middle-man 'driver'
object.

This module also attempts a first use of the Catch test framework for Falaise
modules. All unit tests address the algorithm only, not the Falaise module
structure nor the embedding of this module into flreconstruct.

## Cluster process:

Two filters simplify clustering tasks before the main clustering algorithm,
the 3D graphclustering algorithm, processes the data. Algorithm
(A) filters the image of a tracker event as projected in the x-y-plane. This
is a natural projection since the resulting grid is made of the tracker cells
themselves, a 113 by 9 image for each tracker half.

Filtering with (A) involves simply the image label algorithm which seeks to
separate each separately connected structure in the image, see below. The next
filter then introduces the continuous z-coordinate information and attempts to
utilize those measurements for each hit in a labelled cluster. No natural
discretization is offered as with the geiger grid hence a conservative
histogram discretization is attempted. The aim for this algorithm (B), see
below, is simply to detect large gaps or splits in z within previously
labelled structures in order to split them up further.

Once the filters have separated structures in the tracker according to clear
gaps in all three coordinates, one may assume to be left with reasonably
compact and internally completely connectable structures. That assumption
allows to analyse each structure as a graph with well defined spatial
connections between geiger hits, here taken as nodes of the graph. Algorithm
(C) describes such a 3D graphclusterer.

The clustering process idea is to leave clustering as dumb as possible,
i.e. maximise efficiency at the cost of purity. There should be no unclustered
hits left and the set of clusters must contain the true clusters. Increasing
purity, i.e. removing extraneous clusters is left to subsequent processing
at the next stage of reconstruction.

Dumbness of the algorithm here refers to a minimal set of assumptions, for
instance no information on geometry other than the extent of the tracker image
is required. The clustering is indifferent to lines or helices or any other
structure properties. Such physics information is deferred to the next stage
of processing, for instance using fitting of models to structures.

## Algorithms: (A) Connected structure search

As the name suggests, this cluster algorithm considers the tracker geiger
cells as pixels of an image, a 2 by 9 by 113 pixel image to be precise. Since
the demonstrator tracker will hardly be ripped apart in the near future, this
arrangement of geiger cells is fixed in the code and not considered to be
configurable.

This first algorithm targets the simplest structures to cluster in an event,
checks that they are indeed simple and proceeds or hands over the entire event
to a second algorithm filtering the structures further. It does so for
each tracker half separately. The image under consideration is hence a 9
columns times 113 rows pixel image.

The first method for image segmentation clustering uses purely an image
labeling algorithm. Each connected set of pixels (geiger cells) forms a single
structure and is considered a single cluster. This works fine for single
tracks and helices and such like. The advantage is that no assumptions
regarding any other properties (straight, curved, length, etc.) are
required. Pixels are connected if they are neighbours. In our rectangular grid
tracker, each pixel can have eight neighbours apart from the perimeter pixels.

The disadvantage is that no assumptions are made hence N tracks coming
together at some random points will be considered connected and become a
single cluster. That is obviously wrong. The method considers the simplest
possible split of a singly connected structure before handing over to other
methods for anything else. This, together with simple, disconnected
structures, covers the majority of tracker events and is correspondingly
swift.

For splitting structures, the only consideration is on splits
from one to many tracks, e.g. a V-shape event, for instance due to two
electrons from the same vertex. They can run through the tracker together for
quite some length and split up in two separate tracks at any point. Separate
here means pixels are separated by at least one pixel in the Off state, i.e. a
geiger cell that did not fire.

The method to find the split proceeds as follows: Slice the
half-tracker image in two parts, determine the number of structures in each
separately, then move the cut and repeat. Cutting here take place along the
9 columns of the tracker. If at any point the number of structures on the left is
different to the number of structures on the right then a split has
occurred. Any split has to be handled by the graph cluster
algorithm.

Faulty geiger cells are considered to be set in an On state, i.e. always
present as calibrated geiger hits for the algorithm to work and not create
artificial gaps in cluster structures. The clusters are always overcomplete
which means they can't loose any true cluster hits but can contain more than
strictly needed.

The smartness of making physics sense of the collection of geiger hits in a
cluster is assumed to be introduced at a later stage in the reconstruction
chain. This clustering algorithm as the first stage tries to be as dumb as
possible but with the certainty of not loosing any true hits under any
circumstances.

## Algorithms: (B) ZClusterer

A preliminary algorithm to utilize the z coordinate information in aid of
imagesegmentation. Should be simple to be robust. The current version serves
to finish off the previous clustering in (A) as pre-filter. This algorithm
seeks to find gaps in the collection of z-coordinate values for each labelled
cluster.

Lacking a natural discretization as in the geiger cell grid, this
algorithm uses a histogram discretization of z-values in order to detect such
splits between values. This is attempted to be as safe as possible in order
not to split structures unnecessarily.

## Algorithms: (C) 3D Graph clustering

Arriving at more complex clustering tasks, the same assumption of simplicity
prevails, i.e. try not to be too smart but make certain not to loose the true
clusters. The price to pay is to allow many redundant or non-sensical clusters
to be formed and get rid of at a later stage in the reconstruction. Track
fitting for instance would do that quite easily as bad clusters are bound to
give bad fits.

This algorithm cost is exacerbated in this case of complex structures. If it
is not clear due to branching and randomly stopping tracks where something
starts or stops or splits or bounces then the only safe option is to allow for
all cases hence form a lot of clusters.

A Graph as a data structure is ideal for holding data points and relations
between them. One can then easily reveal data relations using simple, standard
methods for graphs. The crucial point for this choice is that very few
assumptions are required to enter at this stage, none relating to geometry for
instance. Clustering can be made geometry agnostic and hence cover all cases
for tracker structure clustering without bias, i.e. stay as dumb as possible.

Transforming the tracker data into an abstract graph structure works as
follows: (a) Consider each triggered geiger cell as a node in the graph and
set up criteria for connectedness, i.e. when does one node count as connected
to another. In 2D with the geiger cell grid that is quite easy but for a 3D
node, z-information has to enter. Here a neighbour to a cell as to be one of
the 8 grid neighbours and less than a maximum z distance away where the latter
results from a survey of all sorted z coordinate values. It is at this stage
that the z-filtering to find structure splits in z is important. The graph
creation relies on working on a compact structure in all three coordinates.

(b) Edges for the graph are then created between nodes with the help of a
kdTree structure which is ideal to deliver nearest neighbours to a given point
or data member. The collection of edges then creates the graph in 3D.

Once there is a complete graph structure of a tracker event, the algorithm
tries to determine start and end nodes of all possible shortest paths for that
graph. This is the point where clusters are formed hence it is vital not to
loose sensible start and end points. Taking all means forming a lot of paths
which are then each turned into a cluster. However, it also means the true
clusters have to be in the set of solutions. Defining start and end points is
easy when trying to capture all, i.e. every node on the periphery of the tracker
grid is a candidate as is every singly connected node, i.e. a node with a
single edge attached. The latter captures all cases where tracks start/end anywhere
inside the tracker grid.

Caution at the start-end point formation is important. Events in the tracker
render the notion of what is a start and what an end meaningless. Tracks can
come from anywhere and even curve such that some boundaries are never
touched. For completeness hence all permutations of starts and ends including
'dead-ends' are permitted.

However one further tracker property has to be taken into account here as a
final point. Leaving individual geiger cells as autonomous entities at this
stage would be wrong. The geiger cell tracker can simply not measure whether
neighbouring(!) cells were triggered by a single or multiple particles. When
forming clusters, each clusters is supposed to belong to a single particle
hence neighbouring geiger cells should keep that feature of being
neighbours and allowed to show up in several clusters.

This is taken account of by performing one-dimensional clustering to abstract
clumps of geiger hits into countable nodes, called lumped nodes in the
code. Here every one of the 9 tracker
columns in each tracker-half is clustered separately into clumps of
neighbouring rows of activated geiger hits.

While working with individual geiger cells as normal nodes, each node is
replaced in a final step by their corresponding lumped node of which they are
a unique member. This is particularly important for determining start and end
points for the cluster creation from paths in the graph.

Forming a shortest path from start to end then is a purely abstract operation
between nodes, i.e. no notion of geometry enters other than that nodes are neighbours
hence are connected. The clustering doesn't know anything about
concepts of straight or curved or kinked and all edges have the same weight
hence no discrimination of going up or down against horizontal nodes like in a
grid geometry is permitted. The shortest path is hence abstract and the
algorithm delivers all since several possibilities, equally short, can exist
between nodes.

Finishing off, all nodes in each shortest path cluster are transformed back
into collections of tracker pixels and numbered as clusters in a map
container.

## Utilities

The image segmentation library contains three utility objects which might or
might not be of more general interest.

The GG2ImageConverter object takes in geiger hits in the simplest but
sufficient form as a vector of meta
information, where meta information consists of three integers which are
sufficient to uniquely identify a geiger hit, side, row and column. One member
function then converts geiger hits to an image which is stored in the most
compact format as a vector of bools with each bit corresponding to an image
pixel, being either on or off. The idea is to use this function to work on
tracker sides separately hence left and right side tracker images are
delivered.

The second member function, image2gg converts images back to geiger
hit collections, stored in a map with a counting index since these are
designed to be clusters. It is hence not a direct inverse of the first
function from above.

The ImageLabel object does most of the image segmentation work for the image
segmentation algorithm from above, (A). Representing a general image label
algorithm, it offers several convenience functions but works otherwise
identically to, for instance, the label function in the scipy ndimage python
library. Connected pixels are defined as neighbours where neighbours include
pixels on the diagonals.

Finally, the graph object is a specialised graph object structure, chosen to
be as simple as possible since very little is asked of it. Only member
functions delivering simple node properties and a shortest path algorithm are
required. Therefore, the (undirected) graph is specialised to hold only
integer data and keep therefore path and other algorithms as simple as
possible. This is possible since all the more complex data for the image
segmentation is outsourced to a fixed and countable node container outside the
graph object such that the identification of nodes can proceed entirely on the
basis of indices originating from that node container.

The image segmentation library requires only ROOT as external library (for the
kdTree object) as well as C++11 standard libraries for the objects it contains.
