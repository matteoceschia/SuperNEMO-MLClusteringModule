#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
//to pass things to Python
#include "boost/python/detail/wrap_python.hpp"
#include "numpy/arrayobject.h"

// Ourselves:
#include <mlclustering_module.h>

// Standard library:
#include <stdexcept>
#include <iostream>
#include <fstream>

//library to import CNN
#include <fdeep/fdeep.hpp>

// falaise
#include <falaise/snemo/datamodels/data_model.h>

// Registration instantiation macro :
DPP_MODULE_REGISTRATION_IMPLEMENT(mlclustering_module,
				  "mlclustering_module")

void mlclustering_module::initialize(const datatools::properties  & setup_,
					       datatools::service_manager   & service_manager_,
					       dpp::module_handle_dict_type & /* module_dict_ */)
{
  DT_THROW_IF (this->is_initialized(),
	       std::logic_error,
	       "Module 'MLClustering' is already initialized ! ");

  dpp::base_module::_common_initialize(setup_);

  // check the label
  _TCD_label_ = snemo::datamodel::data_info::default_tracker_clustering_data_label();

  eventCounter = 0;
  this->_set_initialized(true);

  return;
}

void mlclustering_module::reset()
{
  DT_THROW_IF (! this->is_initialized(),
	       std::logic_error,
	       "Module 'MLClustering' is not initialized !");
  this->_set_initialized(false);

  // clean up
  _TCD_label_.clear();

  eventCounter = 0;
  std::cout << "Machine learning clustering finished." << std::endl;
  return;
}

// Constructor :
mlclustering_module::mlclustering_module(datatools::logger::priority logging_priority_)
  : dpp::base_module(logging_priority_)
{
}

// Destructor :
mlclustering_module::~mlclustering_module()
{
  // MUST reset module at destruction
  if (this->is_initialized()) reset();
}

// Processing :
dpp::base_module::process_status mlclustering_module::process(datatools::things & data_record_)
{
  DT_THROW_IF (! this->is_initialized(), std::logic_error,
	       "Module 'MLClustering' is not initialized !");

  ///////////////////////////
  // Check calibrated data //
  ///////////////////////////

  // Check if some 'calibrated_data' are available in the data model:
  if (!data_record_.has("CD")) {
    std::cerr << "failed to grab CD bank " << std::endl;
    return dpp::base_module::PROCESS_INVALID;
  }

  // grab the 'calibrated_data' entry from the data model :
  const snemo::datamodel::calibrated_data& the_calibrated_data
    = data_record_.get<snemo::datamodel::calibrated_data>("CD");
  const snemo::datamodel::calibrated_data::tracker_hit_collection_type& gg_hits
    = the_calibrated_data.calibrated_tracker_hits();


  ///////////////////////////////////
  // Check tracker clustering data //
  ///////////////////////////////////

  bool preserve_former_output = false;

  // check if some 'tracker_clustering_data' are available in the data model:
  snemo::datamodel::tracker_clustering_data * ptr_cluster_data = 0;
  if (! data_record_.has(_TCD_label_)) {
    ptr_cluster_data = &(data_record_.add<snemo::datamodel::tracker_clustering_data>(_TCD_label_));
  } else {
    ptr_cluster_data = &(data_record_.grab<snemo::datamodel::tracker_clustering_data>(_TCD_label_));
  }
  snemo::datamodel::tracker_clustering_data & the_clustering_data = *ptr_cluster_data;
  if (the_clustering_data.has_solutions())
    if (! preserve_former_output)
      the_clustering_data.reset();


  /********************
   * Process the data *
   ********************/

  // Main processing method :
	//load CNN model
	const auto model = fdeep::load_model("./fdeep_model.json");

	// Process the clusterizer :
  namespace sdm = snemo::datamodel;

  // make a clustering solution
  sdm::tracker_clustering_solution::handle_type htcs(new sdm::tracker_clustering_solution);
  the_clustering_data.add_solution(htcs, true);
  the_clustering_data.grab_default_solution().set_solution_id(the_clustering_data.get_number_of_solutions() - 1);
  sdm::tracker_clustering_solution & clustering_solution = the_clustering_data.grab_default_solution();

  // Process geiger hits for clustering
  std::cout << "In process: event counter = " << eventCounter << std::endl;

	//vectors for rows, columns and side
	std::vector<int> rows_left;
	std::vector<int> columns_left;
	std::vector<double> zs_left;
	std::vector<int> rows_right;
	std::vector<int> columns_right;
	std::vector<double> zs_right;
	//input tensor
	fdeep::tensor input_left = fdeep::tensor(fdeep::tensor_shape(113, 9, 1), 0);
	fdeep::tensor input_right = fdeep::tensor(fdeep::tensor_shape(113, 9, 1), 0);

  // geiger hits to image
  bool delayed = false;
  MetaInfo mi;
  std::vector<MetaInfo> gg_data; // divide in prompt and delayed hits
  std::vector<MetaInfo> gg_data_delayed;
  for (const sdm::calibrated_data::tracker_hit_handle_type& gg_handle : gg_hits) {
    if (! gg_handle) continue;
    const sdm::calibrated_tracker_hit & snemo_gg_hit = gg_handle.get();
    mi.side   = snemo_gg_hit.get_geom_id().get(1); // 0, 1
    mi.row    = snemo_gg_hit.get_geom_id().get(3); // 113
    mi.column = snemo_gg_hit.get_geom_id().get(2); // 9
    mi.z      = snemo_gg_hit.get_z(); // geiger z value

    if (snemo_gg_hit.is_prompt()){
      gg_data.push_back(mi);
			if(mi.side == 0){
				rows_left.push_back(mi.row);
				columns_left.push_back(mi.column);
				zs_left.push_back(mi.z);
				unsigned int row_pos = mi.row;
				unsigned int column_pos = mi.column;
				const fdeep::tensor_pos pos{row_pos, column_pos,0};
				input_left.set(pos, 255.);
			}
			else{
				rows_right.push_back(mi.row);
				columns_right.push_back(mi.column);
				zs_right.push_back(mi.z);
				unsigned int row_pos = mi.row;
				unsigned int column_pos = mi.column;
				const fdeep::tensor_pos pos{row_pos, column_pos,0};
				input_right.set(pos, 255.);
			}

			/*
			//tensor pos to insert
			unsigned int row_pos = mi.row;;
			unsigned int column_pos;
			if(mi.side == 0){
					column_pos = mi.column;}
			else
			    column_pos = mi.column + 9;
			const fdeep::tensor_pos pos{row_pos, column_pos,0};
			input.set(pos, 1.);
			*/
		}
    else
      gg_data_delayed.push_back(mi);
  }
  //  std::cout << "In process: gg_data size=" << gg_data.size() << std::endl;
  //  std::cout << "In process: gg_data_delayed size=" << gg_data_delayed.size() << std::endl;
  // std::cout <<  "SHOW TENSOR " << fdeep::show_tensor(input_left)<<std::endl;
	//number of clusters
  int nClus_left;
	//initiate C++ labels
	int* c_labels_left;

	//initiate clustering only if hits > 1
	if (rows_left.empty()){nClus_left=0;}
	else if (rows_left.size()==1){nClus_left=1;}
	else if (rows_left.size()>1){

      // std::cout <<  "SHOW TENSOR " << fdeep::show_tensor(input_left)<<std::endl;
			//compute result from CNN
      auto result = model.predict({input_left});
      //std::cout <<  "SHOW TENSOR res" << fdeep::show_tensor(result.at(0))<<std::endl;
			//convert it to "manageable" object
			std::vector<fdeep::tensor> v = result;
      const std::vector<float> vec = *v.at(0).as_vector();
			//get number of clusters
      nClus_left = std::distance(vec.begin(), max_element(vec.begin(), vec.end()));
      if(nClus_left==0){nClus_left=1;}
			std::cout<< "NUMBER OF LEFT CLUSTERS IS "<<nClus_left<<std::endl;
			// for(int i=0; i<vec.size();i++){
			// 	std::cout<<"PRED LABELS "<< vec.at(i) <<std::endl;
			// }

			//send coord to Python for clustering
			Py_Initialize();
			_import_array();
			PyRun_SimpleString("import sys");
			//!!need exact path of where the py_module is!
		  PyRun_SimpleString("sys.path.append(\".\")");
		  PyObject *pName, *pModule;
			//std::cout<< "NUMBER OF LEFT CLUSTERS BEFORE PASSING IS "<<nClus_left<<std::endl;
			PyObject* pNclus = PyLong_FromLong(nClus_left);

			const char *module_name = "py_clustering_module";
		  pName = PyUnicode_FromString(module_name);
			pModule = PyImport_Import(pName);
			Py_DECREF(pName);

			if (pModule != NULL) {
			    const char *func_name = "cluster";
		    	PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
		      if (pFunc && PyCallable_Check(pFunc)){
		          //copy vector to array
							int* arrRows = &rows_left[0];
							int* arrColumns = &columns_left[0];
							double* arrZs = &zs_left[0];
							const int SIZE_ARR = rows_left.size();

							//set dimensions for np.array
						  const int NDIM = 1;
				      npy_intp dim[NDIM]{SIZE_ARR};

							//create python PyObjects
							PyObject *pArrayRows = PyArray_SimpleNewFromData(NDIM, dim, NPY_INT, reinterpret_cast<void*>(arrRows));
			        PyObject *pArrayColumns = PyArray_SimpleNewFromData(NDIM, dim, NPY_INT, reinterpret_cast<void*>(arrColumns));
							PyObject *pArrayZs = PyArray_SimpleNewFromData(NDIM, dim, NPY_DOUBLE, reinterpret_cast<void*>(arrZs));

							PyObject *pLabels = PyObject_CallFunctionObjArgs(pFunc, pNclus, pArrayRows, pArrayColumns, pArrayZs, NULL);
              Py_DECREF(pNclus);
							if (!pLabels)
		              std::cout<<"Fail call"<<std::endl;
		          if (!PyArray_Check(pLabels))
		              std::cerr << "did not return an array" << std::endl;

					    PyArrayObject *np_labels = reinterpret_cast<PyArrayObject*>(pLabels);

							//convert to C++ object
		          c_labels_left = static_cast<int*>(PyArray_DATA(np_labels));
							int len_labels_left = PyArray_SHAPE(np_labels)[0];
/*
							std::cout<<"PRINT LABELS RIGHT IN C++"<<std::endl;
							for (int i = 0; i < len_labels_left; i++){
                std::cout<< c_labels_left[i] << " ";
	    					std::cout << std::endl;
								}
*/
							Py_DECREF(pLabels);
						  Py_DECREF(pArrayRows);
							Py_DECREF(pArrayColumns);
							Py_DECREF(pArrayZs);


					}

			//Py_DECREF(pLabels);
		  //Py_DECREF(pArrayRows);
			//Py_DECREF(pArrayColumns);
			Py_XDECREF(pFunc);
			Py_DECREF(pModule);
		  }
	}


	//number of clusters
  int nClus_right;
	//initiate C++ labels
	int* c_labels_right;

	//initiate clustering only if hits > 1
	if (rows_right.empty()){nClus_right=0;}
	else if (rows_right.size()==1){nClus_right=1;}
	else if (rows_right.size()>1){

			//compute result from CNN
      auto result = model.predict({input_right});

			//convert it to "manageable" object
			std::vector<fdeep::tensor> v = result;
      const std::vector<float> vec = *v.at(0).as_vector();
			//get number of clusters
      nClus_right = std::distance(vec.begin(), max_element(vec.begin(), vec.end())) +1;
			std::cout<< "NUMBER OF RIGHT CLUSTERS IS "<<nClus_right<<std::endl;
			if(nClus_right==0){nClus_right=1;}
			//send coord to Python for clustering
			Py_Initialize();
			_import_array();
			PyRun_SimpleString("import sys");
			//!!need exact path of where the py_module is!
		  PyRun_SimpleString("sys.path.append(\".\")");
		  PyObject *pName, *pModule;
			//std::cout<< "NUMBER OF RIGHT CLUSTERS BEFORE PASSING IS "<<nClus_right<<std::endl;
			PyObject* pNclus = PyLong_FromLong(nClus_right);

			const char *module_name = "py_clustering_module";
		  pName = PyUnicode_FromString(module_name);

			pModule = PyImport_Import(pName);
			Py_DECREF(pName);

			if (pModule != NULL) {
			    const char *func_name = "cluster";
		    	PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
		      if (pFunc && PyCallable_Check(pFunc)){
		          //copy vector to array
							int* arrRows = &rows_right[0];
							int* arrColumns = &columns_right[0];
							double* arrZs = &zs_right[0];
							const int SIZE_ARR = rows_right.size();

							//set dimensions for np.array
						  const int NDIM = 1;
				      npy_intp dim[NDIM]{SIZE_ARR};

							//create python PyObjects
							PyObject *pArrayRows = PyArray_SimpleNewFromData(NDIM, dim, NPY_INT, reinterpret_cast<void*>(arrRows));
			        PyObject *pArrayColumns = PyArray_SimpleNewFromData(NDIM, dim, NPY_INT, reinterpret_cast<void*>(arrColumns));
							PyObject *pArrayZs = PyArray_SimpleNewFromData(NDIM, dim, NPY_DOUBLE, reinterpret_cast<void*>(arrZs));

							PyObject *pLabels = PyObject_CallFunctionObjArgs(pFunc, pNclus, pArrayRows, pArrayColumns, pArrayZs, NULL);
		          if (!pLabels)
		              std::cout<<"Fail call"<<std::endl;
		          if (!PyArray_Check(pLabels))
		              std::cerr << "did not return an array" << std::endl;

					    PyArrayObject *np_labels = reinterpret_cast<PyArrayObject*>(pLabels);
              int len_labels_right = PyArray_SHAPE(np_labels)[0];
							//convert to C++ object
		          c_labels_right = static_cast<int*>(PyArray_DATA(np_labels));

/*
							std::cout<<"PRINT LABELS RIGHT IN C++"<<std::endl;
							for (int i = 0; i < len_labels_right; i++){
                std::cout<< c_labels_right[i] << " ";
	    					std::cout << std::endl;
								}
*/
							Py_DECREF(pLabels);
						  Py_DECREF(pArrayRows);
							Py_DECREF(pArrayColumns);
							Py_DECREF(pArrayZs);
							Py_DECREF(pNclus);
					}

			//Py_DECREF(pLabels);
		  //Py_DECREF(pArrayRows);
			//Py_DECREF(pArrayColumns);
			Py_XDECREF(pFunc);
			Py_DECREF(pModule);
		  }
	}


	// Library objects for clustering
  GG2ImageConverter g2i(18,113); // full sized tracker, 113 rows at 9 columns for 2 sides

  // filter method (A)
  ImageLabel ilab(9,113); // for prompt hits
  ImageLabel ilab_d(9,113); // for delayed hits

  // filter method (B)
  ZClusterer clclean; // splitting in z

  // clusterer
  GraphClusterer3D gcl(9,113);
  gcl.setZResolution(20.0); // [mm]

  // Prompt hits, Left Tracker
  //**************************
  if (gg_data.size()>0) { // work on prompt hits
    // for (MetaInfo& entry : gg_data)
    //   std::cout << "Cluster Entry: (" << entry.side << ", " << entry.column << ", " << entry.row << ")" << std::endl;
    g2i.gg2image(gg_data);
    std::vector<bool> ll = g2i.getLeft();
    std::vector<bool> rr = g2i.getRight();

///////////

///////////

		std::ofstream output_file("./example.txt");
		std::vector<int> ll_copy;
		for(int i=0; i<ll.size();i++){
			ll_copy.push_back(int(rr.at(i)));
			output_file << ll_copy[i] << std::endl;
		}

		//for (int n=0; n<vsize; n++)
		//	{
    //		myfile << returns[n] << endl;
		//	}

    //std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    //std::copy(ll_copy.begin(), ll_copy.end(), output_iterator);

/*
    // pre-filter
    ilab.label(ll); // left side
    std::unordered_map<unsigned int, std::list<Pixel> > labels_left = ilab.getLabels();
    ilab.label(rr); // right side
    std::unordered_map<unsigned int, std::list<Pixel> > labels_right = ilab.getLabels();
    std::cout << "In process: label left cluster size= " << labels_left.size() << std::endl;
    std::cout << "In process: label right cluster size= " << labels_right.size() << std::endl;
*/
/*
		std::cout << "PRINT LABELS LEFT "<<std::endl;
		for ( auto it = labels_left.begin(); it != labels_left.end(); ++it ){
				for (auto v : it->second){
					std::cout << " " << it->first << " : " << v.first <<"," << v.second;}
  	std::cout << std::endl;}
*/

		std::unordered_map<unsigned int, std::vector<MetaInfo> > mymap_left;
		mymap_left.reserve(nClus_left);
		for(unsigned int i = 1; i<nClus_left+1;i++){
			std::vector<MetaInfo> clus_list;
			for(int j = 0; j<rows_left.size(); j++){
				if(c_labels_left[j]==i){
					MetaInfo mi;
					mi.column = columns_left.at(j);
					mi.row = rows_left.at(j);
					mi.z = zs_left.at(j);
					mi.side = 0;
					clus_list.push_back(mi);
				}
			}
			mymap_left[i] = clus_list;
		}



		std::unordered_map<unsigned int, std::vector<MetaInfo> > mymap_right;
		mymap_right.reserve(nClus_right);
		for(unsigned int i = 1; i<nClus_right+1;i++){
			std::vector<MetaInfo> clus_list;
			for(int j = 0; j<rows_right.size(); j++){
				if(c_labels_right[j]==i){
					MetaInfo mi;
					mi.column = columns_right.at(j);
					mi.row = rows_right.at(j);
					mi.z = zs_right.at(j);
					mi.side = 1;
					clus_list.push_back(mi);
				}
			}
			mymap_right[i] = clus_list;
		}

		if (mymap_left.size()>0) {
			// store in clustering solution
		_translate(the_calibrated_data, clustering_solution, mymap_left, delayed);
		}

		if (mymap_right.size()>0) {
			// store in clustering solution
		_translate(the_calibrated_data, clustering_solution, mymap_right, delayed);
		}



/*
		std::cout << "PRINT MY LABELS LEFT, size "<< mymap_left.size()<<std::endl;
 	 for ( auto it = mymap_left.begin(); it != mymap_left.end(); ++it ){
 			 for (auto v : it->second){
 				 std::cout << " " << it->first << " : " << v.first <<"," << v.second;}
 	 std::cout << std::endl;}
*/


/*
    // back to meta info
    if (labels_left.size()>0) {
      // pre-filter on z splits
      std::unordered_map<unsigned int, std::vector<MetaInfo> > label_cls_left = g2i.image2gg(gg_data, labels_left, 0);
      clclean.init(label_cls_left);
      clclean.setZResolution(20.0); // [mm] z resolution
      clclean.zSplitter(); // find z split
      label_cls_left = clclean.getClusters(); // overwrite collection
      std::cout << "In process: after clclean, left cluster size=" << label_cls_left.size() << std::endl;

      // for each pre clustered
      gcl.cluster(label_cls_left);
      std::unordered_map<unsigned int, std::vector<MetaInfo> > clusters_ll = gcl.getClusters();
      std::cout << "In process: after graph3D, left cluster size=" << clusters_ll.size() << std::endl;
      if (clusters_ll.size()>0) {
	// store in clustering solution
	_translate(the_calibrated_data, clustering_solution, clusters_ll, delayed);
      }
    }

    // Right Tracker
    //***************************
    if (labels_right.size()>0) {
      std::unordered_map<unsigned int, std::vector<MetaInfo> > label_cls_right = g2i.image2gg(gg_data, labels_right, 1);
      clclean.init(label_cls_right);
      clclean.zSplitter(); // find z split
      label_cls_right = clclean.getClusters(); // overwrite collection
      std::cout << "In process: after clclean, right cluster size=" << label_cls_right.size() << std::endl;

      gcl.cluster(label_cls_right);
      std::unordered_map<unsigned int, std::vector<MetaInfo> > clusters_rr = gcl.getClusters();
      std::cout << "In process: after graph3D, right cluster size=" << clusters_rr.size() << std::endl;
      if (clusters_rr.size()>0) {
	// store in clustering solution
	_translate(the_calibrated_data, clustering_solution, clusters_rr, delayed);
      }
  	}
*/

  }

  // Delayed hits, Left Tracker
  //***************************
  if (gg_data_delayed.size()>0) { // work on delayed hits
    delayed = true;
    g2i.gg2image(gg_data_delayed);
    std::vector<bool> ll_delayed = g2i.getLeft();
    std::vector<bool> rr_delayed = g2i.getRight();

    // pre-filter
    ilab_d.label(ll_delayed); // left side
    std::unordered_map<unsigned int, std::list<Pixel> > labels_left_d = ilab_d.getLabels();
    ilab_d.label(rr_delayed); // right side
    std::unordered_map<unsigned int, std::list<Pixel> > labels_right_d = ilab_d.getLabels();

    // back to meta info
    if (labels_left_d.size()>0) {
      std::unordered_map<unsigned int, std::vector<MetaInfo> > label_cls_left_d = g2i.image2gg(gg_data_delayed, labels_left_d, 0);
      clclean.init(label_cls_left_d);
      clclean.zSplitter(); // find z split
      label_cls_left_d = clclean.getClusters(); // overwrite collection

      // for each pre clustered
      gcl.cluster(label_cls_left_d);
      std::unordered_map<unsigned int, std::vector<MetaInfo> > clusters_ll_delayed = gcl.getClusters();
      if (clusters_ll_delayed.size()>0) {
	// store in clustering solution
	_translate(the_calibrated_data, clustering_solution, clusters_ll_delayed, delayed);
      }
    }

    // Right Tracker
    //***************************
    if (labels_right_d.size()>0) {
      std::unordered_map<unsigned int, std::vector<MetaInfo> > label_cls_right_d = g2i.image2gg(gg_data_delayed, labels_right_d, 1);
      clclean.init(label_cls_right_d);
      clclean.zSplitter(); // find z split
      label_cls_right_d = clclean.getClusters(); // overwrite collection

      gcl.cluster(label_cls_right_d);
      std::unordered_map<unsigned int, std::vector<MetaInfo> > clusters_rr_delayed = gcl.getClusters();
      if (clusters_rr_delayed.size()>0) {
	// store in clustering solution
	_translate(the_calibrated_data, clustering_solution, clusters_rr_delayed, delayed);
      }
    }
  }
  eventCounter++;
  return dpp::base_module::PROCESS_SUCCESS;
}


void mlclustering_module::_translate(const snemo::datamodel::calibrated_data& the_calibrated_data,
						     snemo::datamodel::tracker_clustering_solution & clustering_solution,
						     std::unordered_map<unsigned int, std::vector<MetaInfo> >& clusters, bool delayed)
{
  namespace sdm = snemo::datamodel;
  MetaInfo mi;
  // translate back to gg hits
  for (auto& entry : clusters) { // loop through map
    // Append a new cluster :
    sdm::tracker_cluster::handle_type tch(new sdm::tracker_cluster);
    if (delayed)
      tch.grab().make_delayed(); // flag this tracker_cluster object as delayed
    else
      tch.grab().make_prompt();

    clustering_solution.grab_clusters().push_back(tch);

    sdm::tracker_cluster::handle_type & cluster_handle
      = clustering_solution.grab_clusters().back();
    // set cluster id number
    cluster_handle.grab().set_cluster_id(clustering_solution.get_clusters().size() - 1);
    // identify all cluster image pixels as geiger hits
    for (MetaInfo& val : entry.second) { // loop over std::vector
      //      std::cout << "translate: Cluster Key: " << entry.first <<" Entry: (" << val.side << ", " << val.column << ", " << val.row << ", " << val.z << ")" << std::endl;
      for (const sdm::calibrated_data::tracker_hit_handle_type& gg_handle : the_calibrated_data.calibrated_tracker_hits()) {
	if (! gg_handle) continue;
	const sdm::calibrated_tracker_hit & snemo_gg_hit = gg_handle.get();
	mi.side   = snemo_gg_hit.get_geom_id().get(1);
	mi.row    = snemo_gg_hit.get_geom_id().get(3);
	mi.column = snemo_gg_hit.get_geom_id().get(2);
	mi.z      = snemo_gg_hit.get_z();
	// check coordinates for identification and resolution interval around z
	if (val.side==mi.side && val.row==mi.row && val.column==mi.column && mi.z>=val.z-5.0 && mi.z<=val.z+5.0) {
	  cluster_handle.grab().grab_hits().push_back(gg_handle); // found, store in cluster
	}
      }
    }
  }
  return;
}
