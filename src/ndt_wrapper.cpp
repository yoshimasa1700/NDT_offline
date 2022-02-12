#include <bits/stdc++.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Dense>

// #define DEBUG  // for store intermidiate values
#include "ndt_scan_matcher.hpp"

namespace np = boost::python::numpy;
namespace p = boost::python;

using namespace std;
using namespace Eigen;


MatrixXd convertNdarrayToEigen(const np::ndarray &n){
  int cols = n.shape(0);
  int rows = n.shape(1);

  float *p = reinterpret_cast<float*>(n.get_data());

  MatrixXd result = MatrixXd::Zero(cols, rows);

  for(int i = 0; i < cols; ++i){
    for(int j = 0; j < rows; ++j){
      result(i, j) = *p;
      p++;
    }
  }

  return result;
}

p::list convertEigenToList(const MatrixXd &src){

  unsigned int r = src.rows();
  unsigned int c = src.cols();

  p::list dest;

  for(unsigned i = 0; i < c; ++i){
    for(unsigned j = 0; j < r; ++j){
      dest.append(src(j, i));
    }
  }

  return dest;
}


class NDT{
public:
  void set_leaf_size(float leaf_size_){
    calc.leaf_size = leaf_size_;
  }

  void create_map(np::ndarray &reference_pc_py){
    MatrixXd cloud = convertNdarrayToEigen(reference_pc_py);
    calc.CreateMap(cloud);
  }

  p::list registration(np::ndarray &scan_pc_py, int max_iteration){
    MatrixXd scan_pc = convertNdarrayToEigen(scan_pc_py);

    TransformT initial_trans = MatrixXd::Zero(6, 1);

    // convert matrix to pcl.
    TransformT relative_pose = calc.Align(initial_trans, scan_pc, max_iteration);

    return convertEigenToList(relative_pose);
  }

#ifdef DEBUG
  p::list get_jacobian_list(){

    p::list result;

    for(unsigned int i = 0 ; i < calc.jacobian_vector.size(); ++i){
      result.extend(convertEigenToList(calc.jacobian_vector[i]));
    }

    return result;
  }

  p::list get_jacobian_sum_list(){

    p::list result;

    for(unsigned int i = 0 ; i < calc.jacobian_vector_sum.size(); ++i){
      result.extend(convertEigenToList(calc.jacobian_vector_sum[i]));
    }

    return result;
  }

  p::list get_update_list(){
    p::list result;

    for(unsigned int i = 0 ; i < calc.update_vector.size(); ++i){
      result.extend(convertEigenToList(calc.update_vector[i]));
    }

    return result;
  }

  p::list get_hessian_list(){
    p::list result;

    for(unsigned int i = 0 ; i < calc.hessian_vector.size(); ++i){
      result.extend(convertEigenToList(calc.hessian_vector[i]));
    }

    return result;
  }

#endif // DEBUG

  p::list get_map(){
    p::list result;

    for(auto itr = calc.leaves.begin(); itr != calc.leaves.end(); ++itr){

      p::list mean;
      for(int i = 0; i < 3; ++i)
        mean.append(itr->second.mean[i]);

      p::list cov;
      for(int i = 0; i < 6; ++i)
        cov.append(itr->second.params[i]);

      result.append(p::make_tuple(mean ,cov));
    }

    return result;
  }

private:
  Matrix<double, 3, 1> mu;
  Matrix<double, 3, 3> cov;

  NDTCalc calc;
};


BOOST_PYTHON_MODULE(libndt)
{
  Py_Initialize();
  np::initialize();

  p::class_<NDT>("NDT")
    .def("set_leaf_size", &NDT::set_leaf_size)
    .def("create_map", &NDT::create_map)
    .def("get_map", &NDT::get_map)
#ifdef DEBUG
    .def("get_jacobian_list", &NDT::get_jacobian_list)
    .def("get_jacobian_sum_list", &NDT::get_jacobian_sum_list)
    .def("get_update_list", &NDT::get_update_list)
    .def("get_hessian_list", &NDT::get_hessian_list)
#endif // DEBUG
    .def("registration", &NDT::registration);
}
