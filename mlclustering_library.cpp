// std libraries
#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>

// ROOT
#include <TKDTree.h>

// this
#include <mlclustering_library.h>


// *****
// ** Graph clustering methods
// *****
void GraphClusterer3D::cluster(std::unordered_map<unsigned int, std::vector<MetaInfo> >& cls) {
  //***
  // init
  if (finalcls.size())
    finalcls.clear();
  if (clusters.size())
    clusters.clear();
  clusters = cls; // copy to work with
  nodes.clear();
  vertices.clear();
  edges.clear();
  store.clear();
  std::unordered_map<unsigned int, std::vector<MetaInfo> > clustercopy;

  Pixel pp;
  GGHit hit;

  //***
  // prepare internal data using GGHit instead of metainfo 
  std::vector<double> zonly;
  std::vector<std::vector<int> > lumped;
  std::vector<GGHit>::iterator nodeiterator;
  int position=0;
  std::vector<int>::iterator lumpiterator;
  std::vector<int> ggindex;
  std::vector<std::vector<int> > lumpedNodes;
  
  MetaInfo newmi; // replacement meta info for original
  std::vector<MetaInfo> newhits; // contains replacement hits
  
  std::unordered_map<unsigned int, std::vector<MetaInfo> > newcls;
  for (auto& entry : clusters) {
    //    std::cout << "cluster nr. " << entry.first << std::endl;
    nodes.clear();
    edges.clear();
    zonly.clear();
    store.clear();
    vertices.clear();

    std::vector<MetaInfo> cluster = entry.second;
    side = cluster.at(0).side; // same for all metainfos in cluster

    if (cluster.size()<3) {
      clustercopy[clustercopy.size() + 1] = cluster; // rescue
      continue; // next in loop; leave as is
    }

    for (auto& mi: cluster) {
      // fill unique vertices set
      pp.first   = mi.column; // stays integer x=column
      pp.second  = mi.row;    // integer y=row
      hit.first  = pp;       // pixel in first
      hit.second = mi.z;  // z in second
      zonly.push_back(mi.z);
      vertices.insert(hit); // fill vertices set
    }
    //***
    // turn vertices to countable nodes container
    int nn=0;
    for (auto& nd : vertices) {
      //      std::cout << "GGHit Node " << nn << ": "<< nd.first.first << " " << nd.first.second << ", " << nd.second << std::endl;
      nodes.push_back(nd);
      nn++;
    }    
    
    //***
    // set up z-difference search
    std::sort(zonly.begin(), zonly.end());
    double maxdifference=0.0;
    for (int j=1;j<(int)zonly.size();j++) { // all mutual differences
      double slope = zonly[j] - zonly[j-1];
      if (fabs(slope) > maxdifference) maxdifference = fabs(slope);
    }
    //    std::cout << "max z difference: " << maxdifference << std::endl;
    if (maxdifference < 2*zres) maxdifference = 2 * zres; // +- z resolution enforcement

    //***
    // lump nodes in 1D
    // index store of nodes with column map index where store holds node indices
    for (int i=0;i<width;i++) { // loop over tracker columns
      for (int j=0;j<(int)nodes.size();j++)
	if (nodes[j].first.first == i) // gets GGHit.first=>Pixel.first->column
	  ggindex.push_back(j); // all GGHit in column i
      lumpedNodes = cluster1D(ggindex, maxdifference);
      if (lumpedNodes.size())
	store[i] = lumpedNodes;
      ggindex.clear();
    }

    //***
    // make more clusters from input clusters
    // edges require neighbour search with limited z distance
    //    std::cout << "store size: " << store.size() << std::endl;
    if (store.size()<2) {  // just one lumped node - no graph possible
      clustercopy[clustercopy.size() + 1] = cluster; // rescue
      //      std::cout << "store single lumped node." << std::endl;
      continue;  // next in loop; leave as is
    }

    newcls = cluster_withgraph(maxdifference);

    vertices.clear(); // recycle
    //***
    // replace all nodes in clusters with lumped nodes
    // since they  belong together - will create plenty of identical clusters
    // uses vertices set again to have only unique nodes after replacement
    for (auto& cl : newcls) {
      unsigned int clsid = cl.first;
      std::vector<MetaInfo> hits = cl.second;
      for (auto& mi : hits) {
	// make a GGHit for finding
	pp.first   = mi.column; // stays integer x=column
	pp.second  = mi.row;    // integer y=row
	hit.first  = pp;       // pixel in first
	hit.second = mi.z;  // z in second
	
	lumped = store[mi.column]; // which lump of nodes contains this GGHit
	nodeiterator = std::find(nodes.begin(), nodes.end(), hit);
	if (nodeiterator!=nodes.end()) { // found GGHit
	  position = nodeiterator - nodes.begin(); // node index
	  for (auto& nds : lumped) { // nds=vector<int>
	    lumpiterator = std::find(nds.begin(), nds.end(), position);
	    if (lumpiterator!=nds.end()) { // found index in lumped nodes
	      for (int idx : nds)
		vertices.insert(nodes[idx]); // all the lumped nodes, repeats not accepted in set
	    }
	  }
	}
      }
      for (auto& gg : vertices) { // turn GGHits back into new meta info's for replacement
	newmi.side = side;
	newmi.column = gg.first.first;
	newmi.row    = gg.first.second;
	newmi.z      = gg.second;
	newhits.push_back(newmi);
      }
      vertices.clear();
      newcls[clsid] = newhits; // overwrite the current cluster
      newhits.clear();
    }

    for (auto& cl : newcls) {
      std::vector<MetaInfo> hits = cl.second;
      clustercopy[clustercopy.size() + 1] = hits; // store away to safety
    }

    // sweeep leftovers
    bool leftover = true;
    std::vector<MetaInfo> unclustered;
    std::vector<MetaInfo>::iterator hititerator;
    for (auto& nd : nodes) {
      for (auto& cl : clustercopy) {
	std::vector<MetaInfo> hits = cl.second;
	for (auto& minfo : hits) {
	  if (nd.first.first  == minfo.column &&
	      nd.first.second == minfo.row &&
	      nd.second       == minfo.z) {
	    leftover = false;
	    continue;
	  }
	}
      }
      if (leftover) {
	newmi.side = side;
	newmi.column = nd.first.first;
	newmi.row    = nd.first.second;
	newmi.z      = nd.second;
	unclustered.push_back(newmi);
	//	std::cout << "found leftover" << " " << newmi.column << " " << newmi.row << " " << newmi.z << std::endl;
      }
      leftover = true;
    }
    if (!unclustered.empty()) { // add to container
      clustercopy[clustercopy.size() + 1] = unclustered;  // store away to safety
      unclustered.clear();
    }
  } 

  //***
  // finish analysing clusters
  clusters.clear();
  unsigned int counter = 1; // loop counter
  for (auto& cl : clustercopy) {
    std::vector<MetaInfo> hits = cl.second;
    // for (auto& minfo : hits)
    //   std::cout << "in clustercopy - nr " << counter << " " << minfo.column << " " << minfo.row << " " << minfo.z << std::endl;
    clusters[counter] = hits;
    counter++;
  }
  
  remove_copies(); // clear out clusters store of identical copies.
}


std::vector<std::vector<int> > GraphClusterer3D::cluster1D(std::vector<int>& nodeindex, double maxdiff) {
  std::vector<std::vector<int> > lumped;
  if (nodeindex.size()<1) return lumped; // empty case, ensures store has the column

  std::vector<int> nd;
  int row = nodes[nodeindex[0]].first.second; // row of first lumped node
  double z= nodes[nodeindex[0]].second; // z of first lumped node
  for (int& idx : nodeindex) {
    if (fabs(nodes[idx].first.second - row) < 2 && fabs(nodes[idx].second - z) < maxdiff+2*zres) { // nearest neighbour
      nd.push_back(idx); // first entry is trivially in
      row = nodes[idx].first.second; // set to new comparison
      z   = nodes[idx].second;
    }
    else { // distance too large in row or z => next cluster
      lumped.push_back(nd);
      nd.clear();
      nd.push_back(idx);
      row = nodes[idx].first.second;
      z   = nodes[idx].second;
    }
  }
  lumped.push_back(nd); // save the final node collection

//   for (int nn=0;nn<lumped.size();nn++) {
//     std::cout << "lumped nodes nr. " << nn << std::endl;
//     for (int id : lumped[nn])
//       std::cout << "id=" << id << " ";
//     std::cout << std::endl;
//   }

  return lumped;
}



std::unordered_map<unsigned int, std::vector<MetaInfo> > GraphClusterer3D::cluster_withgraph(double maxdiff) {
  // first fill edges container, then use graph
  make_edges(maxdiff); // limited z to make edges; grid x,y is known
  int entries = (int)nodes.size();
  
  Graph gg(entries);
  // fill graph with index integers corresponding to nodes container
  for (const std::pair<GGHit, GGHit> edge : edges) {
    GGHit start_node = edge.first;
    GGHit end_node   = edge.second;
    
    std::vector<GGHit>::iterator it1 = std::find(nodes.begin(), nodes.end(), start_node);
    std::vector<GGHit>::iterator it2 = std::find(nodes.begin(), nodes.end(), end_node);
    
    int start_pos = it1 - nodes.begin(); // index of start node in node container
    int end_pos = it2 - nodes.begin(); // index of end node in node container
    //    std::cout << "fill graph: Edge index of node start: " << start_pos << " index end:" << end_pos << std::endl;
    gg.addEdge(start_pos, end_pos);
  } // graph filled and operative
  
  // get the start and target nodes for the path search
  std::vector<int> dead_ends = all_deadends(gg);
  std::vector<int> starts = column_nodes(gg, 0); // column 0 for starts
  std::vector<int> targets = column_nodes(gg, width-1); // column width-1 for targets
  // for (int idx : dead_ends)
  //   std::cout << "dead_end index: " << idx << " ";
  // std::cout << std::endl;
  // for (int idx : starts)
  //   std::cout << "starts index: " << idx << " ";
  // std::cout << std::endl;
  // for (int idx : targets)
  //   std::cout << "targets index: " << idx << " ";
  // std::cout << std::endl;

  // curving paths to consider
  if (targets.empty()) {
    int whichcol = width-2;
    std::vector<int> nextcolumn = column_nodes(gg, whichcol); // column width-2 for targets
    while (nextcolumn.empty() && whichcol > 0) {
      whichcol--;
      nextcolumn = column_nodes(gg, whichcol); // column whichcol for targets
    }
    if (!nextcolumn.empty())
      targets = nextcolumn;
  }
  if (starts.empty()) {
    int whichcol = 1;
    std::vector<int> nextcolumn = column_nodes(gg, whichcol); // column 1 for starts
    while (nextcolumn.empty() && whichcol < width-1) {
      whichcol++;
      nextcolumn = column_nodes(gg, whichcol); // column whichcol for starts
    }
    if (!nextcolumn.empty())
      starts = nextcolumn;
  }
  // for (int idx : starts)
  //   std::cout << "starts index: " << idx << " ";
  // std::cout << std::endl;
  // for (int idx : targets)
  //   std::cout << "targets index: " << idx << " ";
  // std::cout << std::endl;

  // curving paths to consider again; complete bending inside tracker
  starts.insert(starts.end(), targets.begin(), targets.end());
  targets.insert(targets.end(), starts.begin(), starts.end());

  // merge starts and targets with dead ends
  starts.insert(starts.end(), dead_ends.begin(), dead_ends.end());
  targets.insert(targets.end(), dead_ends.begin(), dead_ends.end());

  std::set<int> pathstarts;
  std::set<int> pathtargets;
  for (int idx : starts)
    pathstarts.insert(idx); // get unique starts
  for (int idx : targets)
    pathtargets.insert(idx); // get unique targets

  // only one from a lumped node
  preventLumped(pathstarts);
  preventLumped(pathtargets);

  // std::cout << "All starts index: ";
  // for (int idx : pathstarts)
  //   std::cout << idx << " ";
  // std::cout << std::endl;
  // std::cout << "All targets index: ";
  // for (int idx : pathtargets)
  //   std::cout << idx << " ";
  // std::cout << std::endl;

  // find paths from starts to targets to form clusters  
  std::list<std::vector<std::vector<int> > > tempCluster;
  for (int s : pathstarts) {
    for (int t : pathtargets) {
      if (gg.isReachable(s, t)) {
	if (s != t) {
	  gg.bfsPaths(s, t);
	  tempCluster.push_back(gg.paths());
	}
      }
    }
  }
  return translate(tempCluster); // fill cls map in suitable format
}
  


void GraphClusterer3D::preventLumped(std::set<int>& indices) {
  // get starter and target indices pointing at nodes in container
  // and clear out all but one that are part of a lumped node from store.
  std::vector<std::vector<int> > lumpedNodes;
  std::vector<int> output; 
  std::set<int>::iterator itset;

  for (auto& entry : store) { // check all lumped node indices
    lumpedNodes = entry.second;
    for (auto& nd : lumpedNodes) { // vector<int>
      output.resize(nd.size()+indices.size()); // max+max output container size
      std::vector<int>::iterator testit = std::set_intersection(nd.begin(), nd.end(), indices.begin(), indices.end(), output.begin());
      output.resize(testit - output.begin());
      if (output.size()>0) { // overlap found
	for (int j=1;j<(int)output.size();j++) {
	  itset = std::find(indices.begin(), indices.end(), output[j]); // got the first in output[0]
	  indices.erase(itset); // remove all others with overlap
	}
      }
    }
  }
}


std::unordered_map<unsigned int, std::vector<MetaInfo> > GraphClusterer3D::translate(std::list<std::vector<std::vector<int> > >& temp) {
  // return the path indexes back into nodes
  // and nodes back into a collection of hits
  // in order to form proper clusters. 

  std::unordered_map<unsigned int, std::vector<MetaInfo> > newcls;
  unsigned int counter = 1; // cluster numbering from 1
  MetaInfo mi;
  GGHit node;
  std::vector<MetaInfo> hits;

  for (auto& entry : temp) { // vector<vector<int>>
    for (auto& path : entry ) { // vector<int>
      //      std::cout << "in translate - key = " << counter << std::endl;
      for (int index : path) {
	node = nodes.at(index); // got the node in a path
	mi.column = node.first.first;  // column in image
	mi.row    = node.first.second; // row in image
	mi.z      = node.second;
	mi.side   = side;
	//	std::cout << "in translate - " << mi.column << " " << mi.row << " " << mi.z << std::endl;
	hits.push_back(mi); 
      } // keep collecting meta infors for every node on path
      newcls[counter] = hits; // got all hits, store as cluster
      hits.clear(); 
      counter++; // ready for next cluster
    }
  } // count up all clusters in the temp collection as distinct clusters.
  return newcls;
}



void GraphClusterer3D::make_edges(double maxdiff) {
  // fill tree with doubles by construction
  int nentries = (int)nodes.size();
  int nneighbours = (nentries>=8) ? 8 : nentries;
  double* x = new double [nentries]; // kdTree needs arrays as input
  double* y = new double [nentries]; // kdTree needs arrays as input
  double* z = new double [nentries]; // kdTree needs arrays as input

  int counter=0;
  for (auto& entry : nodes) { // as stored in nodes
    Pixel pp = entry.first;
    x[counter] = pp.first*44.0 + 22.0; // turn to [mm] regular centrepoint at 22 mm
    y[counter] = pp.second*44.0 + 22.0;    // turn to [mm] in units of cell 44 mm
    z[counter] = entry.second;                  // is already [mm]
    counter++;
  }

  TKDTreeID* hittree = new TKDTreeID(nentries, 3, 1);
  // fill kdTree here
  hittree->SetData(0,x);
  hittree->SetData(1,y);
  hittree->SetData(2,z);
  hittree->Build();
  // kdTree ready to work, make edges for a graph using this

  // turn the kdTree container content into container of edges
  GGHit start; // to hold the starter hit
  GGHit target; // to hold the target hit

  edges.clear();
  double point[3]; // needed for kdTree requests
  double* dist = new double [nneighbours]; // check on nearest 8 neighbours in grid
  int* indx = new int [nneighbours];       // where 8 is the maximum posible in a square grid

  for (counter=0; counter<nentries; counter++) {
    start = nodes.at(counter);
    point[0] = x[counter];
    point[1] = y[counter];
    point[2] = z[counter];
    hittree->FindNearestNeighbors(point,nneighbours,indx,dist); // index needs to correspond to node index
    for (int j=0;j<nneighbours;j++) {
      target = nodes.at(indx[j]);
      if (is_neighbour(start, target, maxdiff)) 
	edges.insert(std::make_pair(start, target)); // only unique edges in set
    }

  }

  delete [] indx;
  delete [] dist;
  delete [] x;
  delete [] y;
  delete [] z;
  delete hittree;
}


bool GraphClusterer3D::is_neighbour(GGHit start, GGHit target, double maxdiff) {
  Pixel ppstart  = start.first;
  Pixel pptarget = target.first;
  double zstart  = start.second;
  double ztarget = target.second;

  if (ppstart.first == pptarget.first && ppstart.second == pptarget.second) // exclude itself as neighbour
    return false;

  // check grid x-y
  if (fabs(ppstart.first-pptarget.first) < 2 && fabs(ppstart.second-pptarget.second) < 2) // one grid place only
    if (fabs(zstart-ztarget) < maxdiff+2*zres)    // closer than max z difference
      return true;
  return false;
}


std::vector<int> GraphClusterer3D::all_deadends(Graph gg) {
  std::vector<int> ends;
  GGHit node;
  for (int index : gg.nodes()) {
    node = nodes.at(index); // translate back to node from container
    if (gg.singleNode(index)) // loose end?
      if (node.first.first != 0 && node.first.first != width-1) // not at the extreme columns
	ends.push_back(index);
  }

  std::vector<int> xwallspecial = check_xwall(gg);
  if (xwallspecial.size()) { // combine with ends
    ends.insert(ends.end(), xwallspecial.begin(), xwallspecial.end());
    // for (int xw : xwallspecial)
    //   std::cout << "from xwall: " << xw << " ";
  }
  //  std::cout << std::endl;
  return ends;
}



std::vector<int> GraphClusterer3D::column_nodes(Graph gg, int col) {
  std::vector<int> found;
  GGHit node;
  for (int index : gg.nodes()) {
    node = nodes.at(index); // translate back to node from container
    if (node.first.first == col)
      found.push_back(index);
  }

  return found;
}


std::vector<int> GraphClusterer3D::check_xwall(Graph gg) {
  std::vector<int> found;
  std::vector<int> found_top;
  std::vector<int> found_bot;
  GGHit node;
  for (int index : gg.nodes()) {
    node = nodes.at(index); // translate back to node from container
    if (node.first.second == 0) // book extreme top row for x-wall
      found_top.push_back(index);
    if (node.first.second == height-1) // book extreme bottom row for x-wall
      found_bot.push_back(index);
  }
  // pattern - triangle challenge - no singly connected node
  // ...OO.. or ....0..
  // ....O..    ...00..
  // for top xwall row 
  int previous;
  std::vector<std::vector<int> > rowcls;
  std::vector<int> nn;
  if (found_top.size()>=1) { // any consecutive neighbours in that row?
    node = nodes.at(found_top[0]);
    previous = node.first.first; // previous column
    for (int idx : found_top) {
      node = nodes.at(idx);
      if (fabs(node.first.first - previous)<2) { // neighbours
	nn.push_back(idx); // minimum one entry always
	previous = node.first.first;
      }
      else {               // not a neighbour anymore
	rowcls.push_back(nn); // new row block
	nn.clear();
      }
    }
    rowcls.push_back(nn); // final row block
  }
  // clear out all non-double consecutive row hits
  for (auto& entry : rowcls) {
    if (entry.size()== 2) {
      found.push_back(entry[0]);
      found.push_back(entry[1]);
    }
    else if (entry.size()== 1)
      found.push_back(entry[0]);
  }

  // for bottom xwall row 
  nn.clear();
  rowcls.clear();
  if (found_bot.size()>=1) { // any consecutive neighbours in that row?
    node = nodes.at(found_bot[0]);
    previous = node.first.first; // previous column
    for (int idx : found_bot) {
      node = nodes.at(idx);
      if (fabs(node.first.first - previous)<2) { // neighbours
	nn.push_back(idx); // minimum one entry always
	previous = node.first.first;
      }
      else {               // not a neighbour anymore
	rowcls.push_back(nn); // new row block
	nn.clear();
      }
    }
    rowcls.push_back(nn); // final row block
  }
  // clear out all non-double consecutive row hits
  for (auto& entry : rowcls) {
    if (entry.size()== 2) {
      found.push_back(entry[0]);
      found.push_back(entry[1]);
    }
    else if (entry.size()== 1)
      found.push_back(entry[0]);
  }

  return found;
}


void GraphClusterer3D::remove_copies() {
  std::set<unsigned int> removal_keys; // only unique entries
  std::set<unsigned int> visited; // only unique entries
  std::vector<Pixel> starter; // temp storage: copies of the std::list
  std::vector<Pixel> target; // in order to be able to sort the container
  std::set<unsigned int>::iterator findit;
  MetaInfo mi;
  Pixel hit; // assess equality only on integer grid as double z is a problem
  for (auto& entry : clusters) { // not in order, so book visited keys
    unsigned int key1 = entry.first; // compare this
    visited.insert(key1);
    for (auto& val : entry.second) { // yields meta info
      hit.first  = val.column;
      hit.second = val.row;
      starter.push_back(hit);
    }
    std::sort(starter.begin(), starter.end());
    
    // std::cout << " visited key " << key1 << " " << std::endl;
    // std::cout << "\nstarter from " << key1 <<std::endl;
    // for (auto& pp : starter) std::cout << "x=" << pp.first << " y=" << pp.second << " ";
    // std::cout << std::endl;
    
    for (auto& comparator : clusters) { // to the rest in the container
      findit = std::find(visited.begin(), visited.end(), comparator.first);

      if (findit == visited.end()) { // not yet compared to as entry
	//	std::cout << " against keys " << comparator.first << " ";
	for (auto& val : comparator.second) { // meta info here
	  hit.first  = val.column;
	  hit.second = val.row;
	  target.push_back(hit);
	} 
	std::sort(target.begin(), target.end());

	// avoid subset equality in equal comparison; only complete equality counts
	if (std::equal(target.begin(), target.end(), starter.begin()) && std::equal(starter.begin(), starter.end(), target.begin())) {
	  removal_keys.insert(comparator.first); // remove equal copies of clusters, only unique keys here
	  //	  std::cout << " booked " << comparator.first << "==" << key1 << " ";
	}
      }
      target.clear();
    }
    starter.clear();
  }
  std::unordered_map<unsigned int, std::vector<MetaInfo> > newcls;
  unsigned int counter = 1;
  for (auto& entry : clusters) {
    findit = std::find(removal_keys.begin(), removal_keys.end(), entry.first);
    if (findit == removal_keys.end()) { // found no key in the removal set
      //      std::cout << " not found " << entry.first << " in removal keys" << std::endl;
      newcls[counter] = entry.second; // book this clusters
      counter++; // new key
    }
  }
  //  std::cout << "cluster size " << clusters.size() << " and after copy removal " << newcls.size() << std::endl;
  for (auto& entry : newcls)  {
    // for (auto& hit : entry.second)
    //   std::cout << "in remove copies - nr " << entry.first << " " << hit.column << " " << hit.row << " " << hit.z << std::endl;
    finalcls[entry.first] = entry.second; // store the cleaned copy
  }
}


// *****
// ** Image Label methods
// *****
void ImageLabel::label(std::vector<bool> data) {
  setImage(data);
  label();
}


std::list<std::vector<bool> > ImageLabel::imagecollection() {
  std::list<std::vector<bool> > collection;
  if (!componentMap.size())
    return collection; // return empty if no labeling has taken place before.

  std::vector<bool> im;
  for (auto& x : componentMap) {
    for (int i=0;i<height;i++)
      for (int j=0;j<width;j++)
	im.push_back(false); // zero image, correct size
  
    std::list<Pixel> pixels = x.second; // extract pixels
    for (auto& pp : pixels) 
      im[pp.second * width + pp.first] = true;

    collection.push_back(im);
    im.clear();
  }
  return collection;
}


std::unordered_map<unsigned int, std::list<Pixel> > ImageLabel::getLabels() {
  std::unordered_map<unsigned int, std::list<Pixel> > countedlabels;
  unsigned int counter = 1; // count labels from 1
  for (auto& x: componentMap) { // make new counted keys for labels
    countedlabels[counter] = x.second;
    counter++;
  }
  return countedlabels;
}


void ImageLabel::setImage(std::vector<bool> data) {
  if (image.size())
    image.clear();
  image = data; // deep copy
}



bool ImageLabel::is_splitting(std::vector<bool> data) {
  std::unordered_map<unsigned int, std::list<Pixel> > dummy_left; // for splits
  std::unordered_map<unsigned int, std::list<Pixel> > dummy_right;

  if (splits.size()) {
    splits.clear();
    splitcolumn.clear();
  }
  bool split = false;
  std::vector<bool> left;
  std::vector<bool> right;
  int nleft, nright;
  int copywidth = width;

  // slice image at column
  for (int cut=1; cut<copywidth; cut++) {
    for (int j=0;j<height;j++)
      for (int i=0;i<cut;i++) 
	left.push_back(data[j*copywidth + i]);
    for (int j=0;j<height;j++)
      for (int i=cut;i<copywidth;i++)
      	right.push_back(data[j*copywidth + i]);
    setImage(left);
    setWidth(cut);
    label();
    nleft = (int)nlabels();
    dummy_left = getLabels();
    setImage(right);
    setWidth(copywidth - cut);
    label();
    nright = (int)nlabels();
    dummy_right = getLabels();
    // fill the splitpoints
    splits.push_back(std::make_pair(nleft, nright)); // counts from 0

    if (nleft>0 && nright>0) { // any data at all?
      if(nleft>nright || nleft<nright) { // not equal somewhere = a split
	split = true;
	splitcolumn.push_back(cut-1); // access container from 0 index
	if (splitcolumn.size()<2) {
	  partial_left = dummy_left; // store in case of single split,
	  partial_right = dummy_right; // first split only
	}
      }
      if(nleft>1 && nright>1) { // both sides too much structure
	split = true;
	splitcolumn.push_back(cut-1); // access container from 0 index
      }
    }
    //    std::cout << "split points: (" << nleft << "," << nright << "); split=" << split << std::endl;
    left.clear();
    right.clear();
    dummy_left.clear();
    dummy_right.clear();
  }
  setWidth(copywidth);
  return split;
}



void ImageLabel::label() {
  componentMap.clear();
  component.clear();
  for (int i = 0; i < width*height; i++)
    component.push_back(i);

  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      {
        _unionCoords(x, y, x+1, y);
        _unionCoords(x, y, x, y+1);
        _unionCoords(x, y, x-1, y+1); // left diagonal
        _unionCoords(x, y, x+1, y+1); // right diagonal
      }
  for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
        {
	  if (!image[y*width + x])
	    continue;
	  int c = y*width + x;
	  while (component[c] != c) c = component[c];
	  
	  pp.first = x; // column
	  pp.second = y; // row
	  componentMap[c].push_back(pp);
        }
    }
}


void ImageLabel::_doUnion(unsigned int a, unsigned int b) {
  // get the root component of a and b, and set the one's parent to the other
  while (component[a] != a)
    a = component[a];
  while (component[b] != b)
    b = component[b];
  component[b] = a;
}


void ImageLabel::_unionCoords(unsigned int x, unsigned int y, unsigned int x2, unsigned int y2) {
  if (y2 < height && x2 < width && image[y*width + x] && image[y2*width + x2])
    _doUnion(y*width + x, y2*width + x2);
}



// *****
// ** Utility Geiger to Image methods
// *****
// half-tracker image 113 x 9 = height x width = rows x columns
void GG2ImageConverter::gg2image(std::vector<MetaInfo> data) {
  if (leftimage.size())
    leftimage.clear();
  if (rightimage.size())
    rightimage.clear();

  for (int i=0;i<height;i++)
    for (int j=0;j<hwidth;j++) {
      leftimage.push_back(false); // zero image, half tracker
      rightimage.push_back(false); // zero image, half tracker
    }
  for (auto& mi : data) {
    int row = mi.row;
    int col = mi.column;
    if (mi.side < 1) // left tracker
      leftimage[row * hwidth + col] = true;
    else
      rightimage[row * hwidth + col] = true;
  }
//   for (int row=0;row<height;row++) {
//     for (int col=0;col<hwidth;col++)
//       std::cout << leftimage.at(row*9+col) << " ";
//     std::cout << std::endl;
//   }

}


std::unordered_map<unsigned int, std::vector<MetaInfo> > GG2ImageConverter::image2gg(std::vector<MetaInfo> data, std::unordered_map<unsigned int, std::list<Pixel> > labels, int side) {

  std::unordered_map<unsigned int, std::vector<MetaInfo> > clusters;

  // temporary storage
  std::vector<MetaInfo> hits;
  MetaInfo hit;

  // cluster of pixels map loop
  for (std::unordered_map<unsigned int, std::list<Pixel> >::iterator it=labels.begin(); it!=labels.end(); ++it) {
    unsigned int key = it->first;
    std::list<Pixel> pixels = it->second;
    for (auto& pp : pixels) { // for every pixel
      // find the geiger hit in data, multiple times if needed
      for (auto& mi : data) {
	if (pp.first == mi.column && pp.second == mi.row && side == mi.side) { // found it
	  hit.side = mi.side;
	  hit.row = mi.row;
	  hit.column = mi.column;
	  hit.z   = mi.z;
	  hits.push_back(hit);
	}
      }
    }
    clusters[key] = hits;
    hits.clear();
  }
  return clusters;
}


// *****
// ** Utility Graph methods
// *****
Graph::Graph(int nv)
{
  V = nv; // v vertices
  MAXINT = 32768; // some big integer distance
  adj.resize(V); // expect V elements
}


void Graph::addEdge(int v, int w)
{
  // for an undirect graph, vertices of an edge are each accessible from another
  adj[v].insert(w); // Add w to v’s list.
  adj[w].insert(v); // and vice versa

}


std::unordered_set<int> Graph::nodes()
{
  std::unordered_set<int> store;
  for (auto& entry : adj) // entry is unordered_set
    for (int value : entry) // integer in set
      store.insert(value); // only unique integers in set
  return store;
}


bool Graph::singleNode(int node)
{
  std::unordered_set<int> partners = adj[node];
  return partners.size()<2;
}


// A BFS based function to check whether d is reachable from s.
bool Graph::isReachable(int s, int d)
{
  
  // Base case
  if (s == d)
    
    return true;
  
  
  
  // Mark all the vertices as not visited
  bool *visited = new bool[V];
  
  for (int i = 0; i < V; i++)
    visited[i] = false;
  
  
  
  // Create a queue for BFS
  std::list<int> queue;
  
  // Mark the current node as visited and enqueue it
  visited[s] = true;
  
  queue.push_back(s);
  
  while (!queue.empty())
    
    {
      
      // Dequeue a vertex from queue and print it
      s = queue.front();
      
      queue.pop_front();
      
      
      // Get all adjacent vertices of the dequeued vertex s
      
      // If a adjacent has not been visited, then mark it visited
      
      // and enqueue it
      for (auto& entry : adj[s])
	{
	  
	  // If this adjacent node is the destination node, then return true
	  if (entry == d)
	    return true;
	  
	  
	  
	  // Else, continue to do BFS
	  if (!visited[entry])
	    {
	      
	      visited[entry] = true;
	      
	      queue.push_back(entry);
	      
	    }
	  
	}
      
    }
  
  return false;
}



void Graph::bfsPaths(int start, int target)
{
  if (!adj.size()) {
    return;
  }
  // clear data members
  allPaths.clear();
  currentPath.clear();
  if (start == target) return; // nothing to do

  std::queue<int> queue;
  std::vector<int> distance(adj.size(), MAXINT);

  // store child-to-parent information
  std::vector<std::unordered_set<int>> prev(adj.size());

  // init
  distance[start] = 0;
  queue.push(start);
  bool ans = false;
  
  while (queue.size()) {
    auto current = queue.front();
    queue.pop();
    
    for (auto child: adj[current]) {
      if (distance[child] == MAXINT) {
	// child node not visited yet
	queue.push(child);
	distance[child] = distance[current] + 1;
	prev[child].insert(current);

      } else if (distance[child] == distance[current] + 1) {
	// multiple child nodes with save distance
	prev[child].insert(current);
      }
      
      if (child == target) {
	ans = true;
      }
    }
  }
  
  if (ans) {
    dfs(prev, target);
    return;
  }

  return;
}



void Graph::dfs(std::vector<std::unordered_set<int>>& prev,
		int node)
{
  currentPath.push_back(node);
  
  // path ends here
  if (prev[node].size() == 0) {
    allPaths.push_back(std::vector<int>(currentPath.rbegin(), currentPath.rend()));
  }
  
  for (auto parent: prev[node]) {
    dfs(prev, parent);
  }

  // backtracking
  currentPath.pop_back();
}



// *****
// ** ZClusterer methods
// *****
void ZClusterer::init(std::unordered_map<unsigned int, std::vector<MetaInfo> >& cls) {
  if (clusters.size())
    clusters.clear();
  clusters = cls; // copy to read from
  clustercopy = cls; // copy to modify
}


void ZClusterer::zSplitter() {
  unsigned int clsid; // starts counting at 1

  for (auto& entry : clusters) { // gives uint, vector<MetaInfo>
    clsid = entry.first;
    zSplit(clsid, entry.second);
  }
}


void ZClusterer::zSplit(unsigned int clsid, std::vector<MetaInfo>& cls) {
  //************************
  // find discontinuity in z
  std::valarray<double> allz(cls.size()); // for finding extremes
  int i=0;
  for (auto& mi : cls) {
    allz[i] = mi.z;
    i++;
  }
  if (i<6) return; // not enough data points to fill rough histogram
  double start = allz.min();
  double end   = allz.max();
  //  std::cout << "zSplit, start z " << start << " end " << end << std::endl;
  
  // Case 1: check all z values on global gap  
  double zlimit = histogramSplit(allz, start, end);

  if (zlimit != DUMMY) {
    //    std::cout << "case 1: zlimit = " << zlimit << std::endl;
    zSplitCluster(clsid, zlimit);
    return;
  }
}



void ZClusterer::zSplitCluster(unsigned int id, double zlimit) {
  // global z split cluster
  std::vector<MetaInfo> cls = clusters[id]; // to be split
  std::vector<MetaInfo> newcls;
  std::vector<int> keepElement;
  
  int i = 0; // counter
  for (auto& mi : cls) {
    double z = mi.z;
    // consistent z values remain in both clusters
    if (z < zlimit + stepwidth && z > zlimit - stepwidth) { // absolute 10mm consistency interval
      newcls.push_back(mi);
      keepElement.push_back(i);
    }
    else if (z <= zlimit - stepwidth) { // outside below zlimit, into newcls
      newcls.push_back(mi);
      //      std::cout << "newcls z = " << z << std::endl;
    }
    else                       // z>=zlimit+stepwidth
      keepElement.push_back(i);
    i++;
  }

  std::vector<MetaInfo> modcls; // dummy storage
  for (int which : keepElement) {
    modcls.push_back(cls.at(which));
    //    std::cout << "modcls z = " << cls.at(which).z << std::endl;
  }

  // avoid single pixels to be split off
  if (keepElement.size() <= 1 || newcls.size() <= 1) {
    return; // do nothing
  }

  // modify cluster collection
  clustercopy[id] = modcls;
  clustercopy[clustercopy.size()+1] = newcls; // keys count from 1
}



double ZClusterer::histogramSplit(std::valarray<double>& allz, double start, double end) {
  // discretize z-axis
  int nbins = floor(fabs(end - start)/100.0)>4 ? floor(fabs(end - start)/100.0-1) : 4; // coarse histogram resolution in z, min. 4
  double step = floor(fabs(end - start) / nbins);
  
  if (fabs((end - start)) / stepwidth <= 8.0) return DUMMY; // z coordinate error size = flat in z, no split

  std::vector<int> histogram(nbins+1, 0); // fill with zero

  // fill histogram
  for (int j=0;j<(int)allz.size();j++) { // for all z
    int bucket = (int)floor((allz[j]-start) / step); // which bin
    histogram[bucket] += 1; // increment
  }

  // check
  // for (int bin : histogram)
  //   std::cout << "histoSplit, bin " << bin << std::endl;
  
  double zlimit = splitFinder(histogram); // find detectable absolute gap in z
  if (zlimit != DUMMY) {
    return zlimit * step + start;
  }
  return DUMMY; // no gap found
}


double ZClusterer::splitFinder(std::vector<int>& hist) {
  int counter=0;
  while (hist[counter]==0 && counter<hist.size()) // find first non-zero entry
    counter++;
  if (counter==hist.size()) return DUMMY; // nothing to do

  std::vector<int>::iterator it;
  it = std::find(hist.begin() + counter, hist.end(), 0); // find first zero after entries

  if (it != hist.end()) { // found a zero in the histogram, a gap
    int pos = it - hist.begin(); // index of gap start in histo

    std::reverse(hist.begin(), hist.end()); // check for zero from the other end
    counter = 0; // again find first non-zero entry
    while (hist[counter]==0) // find first non-zero entry
      counter++;
    it = std::find(hist.begin() + counter, hist.end(), 0); // zero after finite entries
    if (it != hist.end()) {
      int pos2 = hist.size() - (it-hist.begin()); // index of last zero in histo

      double loc = (pos+pos2)/2.0; // average histogram bin position
      return loc; // return z value of average of empty bins as border
    } // no gap between finite entries
    else
      return DUMMY;
  }
  else
    return DUMMY;
}

