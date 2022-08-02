#include <bits/stdc++.h>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Dense>
#include <boost/python/tuple.hpp>

#define DEBUG  // for store intermidiate values
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

  p::list registration(np::ndarray &scan_pc_py, int max_iteration, np::ndarray &initial_trans_py){
    MatrixXd scan_pc = convertNdarrayToEigen(scan_pc_py);

    MatrixXd initial_trans = convertNdarrayToEigen(initial_trans_py);

    cerr << "initial_trans:" << initial_trans.row(0) << endl;

    // TransformT initial_trans = MatrixXd::Zero(6, 1);
    // TransformT initial_trans;
    // initial_trans << 1.79387, 0.720047, 0, 0, 0, 0.6931;
    // initial_trans << 1.79387, 0.720047, 0, 0, 0, 1.4931;

    // convert matrix to pcl.
    TransformT relative_pose = calc.Align(initial_trans.row(0), scan_pc, max_iteration);

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

  p::list get_hessian_sum_list(){

    p::list result;

    for(unsigned int i = 0 ; i < calc.hessian_vector_sum.size(); ++i){
      result.extend(convertEigenToList(calc.hessian_vector_sum[i]));
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

  p::list get_score_list(){
    p::list result;

    for(unsigned int i = 0 ; i < calc.score_vector.size(); ++i){
      result.append(calc.score_vector[i]);
    }

    return result;
  }

  p::list get_found_list(){
    p::list result;

    for(unsigned int i = 0 ; i < calc.neighbor_found_points_.size(); ++i){
      p::list point;

      point.append(calc.neighbor_found_points_[i][0]);
      point.append(calc.neighbor_found_points_[i][1]);
      point.append(calc.neighbor_found_points_[i][2]);

      result.append(point);
    }

    return result;
  }

  p::list get_not_found_list(){
    p::list result;

    for(unsigned int i = 0 ; i < calc.neighbor_not_found_points_.size(); ++i){
      p::list point;

      point.append(calc.neighbor_not_found_points_[i][0]);
      point.append(calc.neighbor_not_found_points_[i][1]);
      point.append(calc.neighbor_not_found_points_[i][2]);

      result.append(point);
    }

    return result;
  }

  p::list query_point(p::list query){
    p::list result;
    float target[3];
    for(int i = 0; i < 3; ++i)
      target[i] = p::extract<double>(query[i]);

    calc.rangeSearchRecursive(target, calc.leaves,
                              calc.nodes_map,
                              calc.root_id,
                              calc.leaf_size);

    for(unsigned int i = 0; i < calc.neighbor_list.size(); ++i){
      int neighbor_id = calc.neighbor_list[i];
      Leaf l = calc.leaves.at(neighbor_id);

      p::list point;
      for(int j = 0; j < 3; ++j)
        point.append(l.mean[j]);
      result.append(point);
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

  p::tuple calc_score(np::ndarray &point_py, np::ndarray &transform_py){
    // get point from ndarray.
    MatrixXd pc = convertNdarrayToEigen(point_py);

    TransformT transform = convertNdarrayToEigen(transform_py);

    PointT point_transformed = TransformPointCloud(pc, transform);

    PointT point = pc;

    // sample gauss param.
    ScoreParams gauss_params(calc.output_prob, calc.leaf_size);

    // calc point jacobian.
    JacobianCoefficientsT jacobian_coefficients;
    HessianCoefficientsT hessian_coefficients;
    angleDerivatives(transform, jacobian_coefficients, hessian_coefficients);

    // calc point hessian.
    PointJacobianT point_jacobian;
    PointHessianT point_hessian;
    pointDerivatives
      (point,
       jacobian_coefficients,
       hessian_coefficients,
       point_jacobian,
       point_hessian);

    // calc x_k_dash
    PointT mean = PointT::Zero();
    CovarianceMatrixT cov = CovarianceMatrixT::Identity();
    CovarianceMatrixT cov_inv;
    bool exists;
    double det = 0;
    cov.computeInverseAndDetWithCheck(cov_inv,det,exists);

    // calc cov inv
    PointT x_k_dash = point_transformed - mean;
    tuple<double, JacobianT, HessianT> iter_derivatives = computeDerivative
      (gauss_params, point_jacobian, point_hessian, x_k_dash, cov_inv);

    p::list jacobian = convertEigenToList(get<1>(iter_derivatives));
    p::list hessian = convertEigenToList(get<2>(iter_derivatives));

    return p::make_tuple(get<0>(iter_derivatives), jacobian, hessian);
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
    .def("calc_score", &NDT::calc_score)
#ifdef DEBUG
    .def("get_jacobian_list", &NDT::get_jacobian_list)
    .def("get_jacobian_sum_list", &NDT::get_jacobian_sum_list)
    .def("get_hessian_sum_list", &NDT::get_hessian_sum_list)
    .def("get_update_list", &NDT::get_update_list)
    .def("get_hessian_list", &NDT::get_hessian_list)
    .def("get_score_list", &NDT::get_score_list)
    .def("get_found_list", &NDT::get_found_list)
    .def("get_not_found_list", &NDT::get_not_found_list)
    .def("query_point", &NDT::query_point)
#endif // DEBUG
    .def("registration", &NDT::registration);
}
