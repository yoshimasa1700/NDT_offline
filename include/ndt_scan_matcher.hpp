#ifndef __NDT_SCAN_MATCHER_HPP__
#define __NDT_SCAN_MATCHER_HPP__

#include <Eigen/src/Core/Matrix.h>
#include <limits>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/Meta.h>
#include <iostream>
#include <map>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <bits/stdc++.h>

using namespace std;
using namespace Eigen;

using TransformT = Matrix<double, 6, 1>;
using JacobianT = TransformT;
using HessianT = Matrix<double, 6, 6>;
using JacobianCoefficientsT = Matrix<double, 8, 3>;
using HessianCoefficientsT = Matrix<double, 18, 3>;
using PointJacobianT = Matrix<double, 3, 6>;
using PointHessianT = Matrix<double, 18, 6>;
using CovarianceMatrixT = Matrix<double, 3, 3>;
using PointT = Matrix<double, 3, 1>;
using PointTT = Matrix<double, 1, 3>;

TransformT init_trans;//イテレーション終了時とコールバックで更新
double transformation_epsilon = 0.01;
double leaf_size = 0.08;
double output_prob = 0.055;

// ros::Publisher vis_pub,vis_pub2;
// ros::Publisher pose_pub,cloud_pub;
int sort_axis=0;
int neighbor_id;
int root_id = -1;

typedef struct
{
  int	id;
  float pos[3];
} point_with_id;

typedef struct
{
  int	parent_id;
  int left_id;
  int right_id;
  int axis;
} node;

typedef struct
{
  int	points = 0;
  // Eigen::Vector3f mean(0, 0, 0);
  float mean[3] = {0,0,0};
  float params[6] = {0,0,0,0,0,0};//11,22,33,12,13,23
  float inverse_params[6]={0,0,0,0,0,0};//11,22,33,12,13,23
  float a[9];
  float eigen_vector[9];
  // Eigen::Matrix3f cov;
  // Eigen::Matrix3f evecs;
  // Eigen::Vector3f evals;
} Leaf;

int AxisSort(const void * n1, const void * n2)
{
  if (((point_with_id *)n1)->pos[sort_axis] > ((point_with_id *)n2)->pos[sort_axis])
	{
      return 1;
	}
  else if (((point_with_id *)n1)->pos[sort_axis] < ((point_with_id *)n2)->pos[sort_axis])
	{
      return -1;
	}
  else
	{
      return 0;
	}
}

int eigenJacobiMethod(float *a, float *v, int n, float eps = 1e-8, int iter_max = 100)
{
  float *bim, *bjm;
  float bii, bij, bjj, bji;

  bim = new float[n];
  bjm = new float[n];

  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      v[i*n+j] = (i == j) ? 1.0 : 0.0;
    }
  }

  int cnt = 0;
  for(;;){
    int i = 0;
    int j = 0;

    float x = 0.0;
    for(int ia = 0; ia < n; ++ia){
      for(int ja = 0; ja < n; ++ja){
        int idx = ia*n+ja;
        if(ia != ja && fabs(a[idx]) > x){
          i = ia;
          j = ja;
          x = fabs(a[idx]);
        }
      }
    }

    float aii = a[i*n+i];
    float ajj = a[j*n+j];
    float aij = a[i*n+j];

    float alpha, beta;
    alpha = (aii-ajj)/2.0;
    beta  = sqrt(alpha*alpha+aij*aij);

    float st, ct;
    ct = sqrt((1.0+fabs(alpha)/beta)/2.0);    // sinθ
    st = (((aii-ajj) >= 0.0) ? 1.0 : -1.0)*aij/(2.0*beta*ct);    // cosθ

    // A = PAPの計算
    for(int m = 0; m < n; ++m){
      if(m == i || m == j) continue;

      float aim = a[i*n+m];
      float ajm = a[j*n+m];

      bim[m] =  aim*ct+ajm*st;
      bjm[m] = -aim*st+ajm*ct;
    }

    bii = aii*ct*ct+2.0*aij*ct*st+ajj*st*st;
    bij = 0.0;

    bjj = aii*st*st-2.0*aij*ct*st+ajj*ct*ct;
    bji = 0.0;

    for(int m = 0; m < n; ++m){
      a[i*n+m] = a[m*n+i] = bim[m];
      a[j*n+m] = a[m*n+j] = bjm[m];
    }
    a[i*n+i] = bii;
    a[i*n+j] = bij;
    a[j*n+j] = bjj;
    a[j*n+i] = bji;

    // V = PVの計算
    for(int m = 0; m < n; ++m){
      float vmi = v[m*n+i];
      float vmj = v[m*n+j];

      bim[m] =  vmi*ct+vmj*st;
      bjm[m] = -vmi*st+vmj*ct;
    }
    for(int m = 0; m < n; ++m){
      v[m*n+i] = bim[m];
      v[m*n+j] = bjm[m];
    }

    float e = 0.0;
    for(int ja = 0; ja < n; ++ja){
      for(int ia = 0; ia < n; ++ia){
        if(ia != ja){
          e += fabs(a[ja*n+ia]);
        }
      }
    }
    if(e < eps) break;

    cnt++;
    if(cnt > iter_max) break;
  }

  delete [] bim;
  delete [] bjm;

  return cnt;
}

int CreateNode(int* root_id,
               int point_size,
               std::vector<node>& nodes,
               std::vector<std::vector<int>> axis_sort_ids,
               int depth,
               int parent_id,
               bool node_is_right)
{
  int group_size = axis_sort_ids[0].size();
  int axis = depth % 3;
  int middle = (group_size - 1) / 2;
  int median_id = axis_sort_ids[axis][middle];

  nodes[median_id].axis = axis;
  nodes[median_id].parent_id = parent_id;
  nodes[median_id].left_id = -1;
  nodes[median_id].right_id = -1;

  if(parent_id >= 0){ // 親あり
    if(!node_is_right) nodes[parent_id].left_id = median_id;
    if(node_is_right) nodes[parent_id].right_id = median_id;
  }
  else{ // 親なし
    *root_id = median_id;
  }

  // have no child
  if(group_size <= 1){
    // std::cout<<"leaf"<<std::endl;
    return 1;
  }

  std::vector<int>::iterator middle_iter(axis_sort_ids[axis].begin());
  std::advance(middle_iter, middle);

  std::vector<int> left_group(axis_sort_ids[axis].begin(), middle_iter);
  ++middle_iter;

  std::vector<int> right_group(middle_iter, axis_sort_ids[axis].end());
  // std::cout<<std::endl;
  // std::cout<<"median_id"<<median_id<<std::endl;
  // std::cout<<"middle"<<middle<<std::endl;
  // std::cout<<"axis"<<nodes[median_id].axis<<std::endl;
  // std::cout<<"group is (";
  // for(int i=0;i<group_size;i++){
  //   std::cout<<axis_sort_ids[axis][i]<<",";
  // }
  // std::cout<<")"<<std::endl;
  // std::cout<<"left_group is (";
  // for(unsigned int i=0;i<left_group.size();i++){
  //   std::cout<<left_group[i]<<",";
  // }
  // std::cout<<")"<<std::endl;
  // std::cout<<"right_group is (";
  // for(unsigned int i=0;i<right_group.size();i++){
  //   std::cout<<right_group[i]<<",";
  // }
  // std::cout<<")"<<std::endl;
  // std::cout<<std::endl;
  std::vector<std::vector<int>> left_axis_sort_ids(3,std::vector<int>(left_group.size()));
  std::vector<std::vector<int>> right_axis_sort_ids(3,std::vector<int>(right_group.size()));

  std::vector<int> next_group(point_size,0);
  std::vector<int> left_axis_count(3,0);
  std::vector<int> right_axis_count(3,0);
  for(unsigned int i = 0; i < left_group.size(); i++){
    left_axis_sort_ids[axis][i] = left_group[i];
    next_group[left_group[i]] = -1;
  }
  for(unsigned int i = 0; i < right_group.size(); i++){
    right_axis_sort_ids[axis][i] = right_group[i];
    next_group[right_group[i]] = 1;
  }
  for(int i = 0; i < group_size; i++){
    for(int j = 0; j < 3; j++){
      if(j==axis) continue;
      if(next_group[axis_sort_ids[j][i]] == -1){
        left_axis_sort_ids[j][left_axis_count[j]] = axis_sort_ids[j][i];
        left_axis_count[j]++;
      }
      else if(next_group[axis_sort_ids[j][i]] == 1){
        right_axis_sort_ids[j][right_axis_count[j]] = axis_sort_ids[j][i];
        right_axis_count[j]++;
      }
    }
  }

  bool left = false;
  bool right = false;
  if(left_group.size() > 0){
    left = CreateNode
      (root_id,
       point_size,
       nodes,
       left_axis_sort_ids,
       depth+1,
       median_id,
       false);
  }else
    left = true;

  if(right_group.size() > 0){
    right = CreateNode
      (root_id,
       point_size,
       nodes,
       right_axis_sort_ids,
       depth+1,
       median_id,
       true);
  }else
    right = true;

  if(right && left)
    return 1;

  return 0;
}

double EuclidDist3D(const float* a, const float* b){
  float d2 = 0;
  for(int i = 0; i < 3; ++i){
    d2 += std::pow(a[i] - b[i], 2);
  }
  return std::sqrt(d2);
}

void searchRecursive(const float* query_position,
                     const std::map <int, Leaf> &leaves,
                     const std::map <int, node> &tree,
                     const int &node_id,
                     double &min_dist){
  // reach leave.
  if(node_id < 0){
    return;
  }
  // std::cout << " node_id = " << node_id;
  auto tree_iter = tree.find(node_id);
  auto leaf_iter = leaves.find(node_id);

  node n = tree_iter->second;
  Leaf l = leaf_iter->second;
  // std::cout << "dist" << std::endl;
  double dist = EuclidDist3D(l.mean, query_position);

  if(dist < min_dist){
    min_dist = dist;
    neighbor_id = node_id;
  }

  int next_id;
  int opp_id;
  // std::cout << "left&right" << std::endl;
  if(query_position[n.axis] < l.mean[n.axis]){
    next_id = n.left_id;///ここ見る
    opp_id = n.right_id;
  }else{
    next_id = n.right_id;///ここ見る
    opp_id = n.left_id;
  }

  searchRecursive(query_position, leaves, tree, next_id, min_dist);

  double diff = std::fabs(query_position[n.axis] - l.mean[n.axis]);
  if (diff < min_dist)
    searchRecursive(query_position, leaves, tree, opp_id, min_dist);
}

Eigen::MatrixXd TransformPointCloud (const Eigen::MatrixXd &cloud,
                                     TransformT tf){
  Eigen::Vector3d trans(tf(0),tf(1),tf(2));
  Matrix3d rot;
  rot = Eigen::AngleAxisd(tf(3), Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(tf(4), Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(tf(5), Eigen::Vector3d::UnitZ());

  return (rot * cloud.transpose()).transpose().rowwise() + trans.transpose();
}

class ScoreParams
{
public:
  ScoreParams(const double &outlier_prob, const double &resolution)
  {
    vector<double> c(2);
    d_ = vector<double>(3);

    // calc c1, c2 in (eq.6.8) [Magnusson 2009]
    c[0] = 10.0 * (1 - outlier_prob);
    c[1] = outlier_prob / pow(resolution, 3);

    d_[2] = -log(c[1]);
    d_[0] = -log(c[0] + c[1]) - d_[2];
    d_[1] = -2 * log((-log(c[0] * exp(-0.5) + c[1]) - d_[2]) / d_[0]);
  }

  double d(const unsigned int i) const{
    return d_[i - 1];
  }
private:
  std::vector<double> d_;
};

tuple<double, JacobianT, HessianT> computeDerivative
(const ScoreParams &param,
 const PointJacobianT &point_jacobian, const PointHessianT &point_hessian,
 const PointT &x_k_dash, const CovarianceMatrixT &cov_inv){

  PointTT xkd_T_conv_inv =  x_k_dash.transpose() * cov_inv;
  double xkd_T_conv_inv_xkd = x_k_dash.dot(xkd_T_conv_inv);
  double exp_term = exp(-param.d(2) / 2 * xkd_T_conv_inv_xkd);
  double d1_d2 = param.d(1) * param.d(2);

  // calc jacobian.
  JacobianT jacobian = xkd_T_conv_inv * point_jacobian;
  jacobian *= (d1_d2 * exp_term);

  // calc hessian.
  HessianT hessian;

  for(unsigned int j = 0; j < 6; ++j){
    for(unsigned int i = 0; i < 6; ++i){

      double t1 = xkd_T_conv_inv * point_jacobian.col(i);
      double t2 = xkd_T_conv_inv * point_jacobian.col(j);
      double t3 = xkd_T_conv_inv * point_hessian.block<3, 1>(i * 3, j);
      double t4 = point_jacobian.col(j).transpose() * cov_inv * point_jacobian.col(i);

      hessian(i, j) = d1_d2 * exp_term *
        (-param.d(2) *
         t1 * t2 + t3 + t4);
    }
  }


  // calc score.
  double score = -param.d(1) * exp_term;

  return make_tuple(score, jacobian, hessian);
}

void angleDerivatives(TransformT & tf,JacobianCoefficientsT & jacobian_coefficients,HessianCoefficientsT & hessian_coefficients){
  double cx, cy, cz, sx, sy, sz;
  // if (fabs(tf(3)) < 10e-5) {
  //   cx = 1.0;
  //   sx = 0.0;
  // } else {
  cx = cos(tf(3));
  sx = sin(tf(3));
  // }
  // if (fabs(tf(4)) < 10e-5) {
  //   cy = 1.0;
  //   sy = 0.0;
  // } else {
  cy = cos(tf(4));
  sy = sin(tf(4));
  // }
  // if (fabs(tf(5)) < 10e-5) {
  //   cz = 1.0;
  //   sz = 0.0;
  // } else {
  cz = cos(tf(5));
  sz = sin(tf(5));
  // }
  jacobian_coefficients <<    (-sx * sz + cx * sy * cz),  (-sx * cz - cx * sy * sz),  (-cx * cy),
    (cx * sz + sx * sy * cz),   (cx * cz - sx * sy * sz),   (-sx * cy),
    (-sy * cz),                 sy * sz,                    cy,
    sx * cy * cz,               (-sx * cy * sz),            sx * sy,
    (-cx * cy * cz),            cx * cy * sz,               (-cx * sy),
    (-cy * sz),                 (-cy * cz),                 0,
    (cx * cz - sx * sy * sz),   (-cx * sz - sx * sy * cz),  0,
    (sx * cz + cx * sy * sz),   (cx * sy * cz - sx * sz),   0;

  hessian_coefficients <<     0,                          0,                          0,
    (-cx * sz - sx * sy * cz),  (-cx * cz + sx * sy * sz),  sx * cy,
    (-sx * sz + cx * sy * cz),  (-cx * sy * sz - sx * cz),  (-cx * cy),
    0,                          0,                          0,
    (cx * cy * cz),             (-cx * cy * sz),            (cx * sy),
    (sx * cy * cz),             (-sx * cy * sz),            (sx * sy),
    0,                          0,                          0,
    (-sx * cz - cx * sy * sz),  (sx * sz - cx * sy * cz),   0,
    (cx * cz - sx * sy * sz),   (-sx * sy * cz - cx * sz),  0,
    (-cy * cz),                 (cy * sz),                  (-sy),
    (-sx * sy * cz),            (sx * sy * sz),             (sx * cy),
    (cx * sy * cz),             (-cx * sy * sz),            (-cx * cy),
    (sy * sz),                  (sy * cz),                  0,
    (-sx * cy * sz),            (-sx * cy * cz),            0,
    (cx * cy * sz),             (cx * cy * cz),             0,
    (-cy * cz),                 (cy * sz),                  0,
    (-cx * sz - sx * sy * cz),  (-cx * cz + sx * sy * sz),  0,
    (-sx * sz + cx * sy * cz),  (-cx * sy * sz - sx * cz),  0;
}

void pointDerivatives(PointT & point,
                      JacobianCoefficientsT & jacobian_coefficients,
                      HessianCoefficientsT & hessian_coefficients,
                      PointJacobianT & point_jacobian,
                      PointHessianT & point_hessian){
  double jacobian_params[8]={0,0,0,0,0,0,0,0};
  vector<Vector3d,aligned_allocator<Vector3d> > hessian_params(6);
  for(int i=0;i<8;i++){
    for(int j=0;j<3;j++){
      jacobian_params[i] += point(j) * jacobian_coefficients(i,j);
    }
  }

  for(int i=0;i<6;i++){
    hessian_params[i] << 0,0,0;
    for(int j=0;j<3;j++){
      hessian_params[i](0) += point(j) * hessian_coefficients(i*3+0,j);
      hessian_params[i](1) += point(j) * hessian_coefficients(i*3+1,j);
      hessian_params[i](2) += point(j) * hessian_coefficients(i*3+2,j);
    }
  }

  point_jacobian = MatrixXd::Zero(3, 6);
  point_hessian = MatrixXd::Zero(18, 6);

  point_jacobian(0,0) = 1;
  point_jacobian(1,1) = 1;
  point_jacobian(2,2) = 1;
  point_jacobian(1, 3) = jacobian_params[0];
  point_jacobian(2, 3) = jacobian_params[1];
  point_jacobian(0, 4) = jacobian_params[2];
  point_jacobian(1, 4) = jacobian_params[3];
  point_jacobian(2, 4) = jacobian_params[4];
  point_jacobian(0, 5) = jacobian_params[5];
  point_jacobian(1, 5) = jacobian_params[6];
  point_jacobian(2, 5) = jacobian_params[7];
  point_hessian.block<3, 1>(9, 3) =  hessian_params[0];
  point_hessian.block<3, 1>(12, 3) = hessian_params[1];
  point_hessian.block<3, 1>(15, 3) = hessian_params[2];
  point_hessian.block<3, 1>(9, 4) =  hessian_params[1];
  point_hessian.block<3, 1>(12, 4) = hessian_params[3];
  point_hessian.block<3, 1>(15, 4) = hessian_params[4];
  point_hessian.block<3, 1>(9, 5) =  hessian_params[2];
  point_hessian.block<3, 1>(12, 5) = hessian_params[4];
  point_hessian.block<3, 1>(15, 5) = hessian_params[5];
}

class NDTCalc{
public:
  void CreateMap(const Eigen::MatrixXd &cloud){
    // reset k-d tree.
    leaves.clear();
    nodes_map.clear();

    // NDT Map Generate
    // search min max point
    float min_p[3] = {std::numeric_limits<float>::max()};
    float max_p[3] = {0};
    for(unsigned int i=0;i<cloud.rows();++i){
      for(int j = 0; j < 3; ++j){
        if(min_p[j]>cloud(i, j))
          min_p[j] = cloud(i, j);
        if(max_p[j]<cloud(i, j))
          max_p[j] = cloud(i, j);
      }
    }

    // calculation min max div voxel
    float inverse_leaf_size = 1 / leaf_size;
    float min_b[3];
    float max_b[3];
    float div_b[3];
    for(int i=0;i<3;i++){
      min_b[i] = static_cast<int>(std::floor(min_p[i] * inverse_leaf_size));
      max_b[i] = static_cast<int>(std::floor(max_p[i] * inverse_leaf_size));
      div_b[i] = max_b[i] - min_b[i] + 1;
    }

    float div_mul[3];
    div_mul[0] = 1;
    div_mul[1] = div_b[0];
    div_mul[2] = div_b[0] * div_b[1];

    for(unsigned int i = 0; i < cloud.rows(); ++i){

      // grid quantization
      int axis_v_index[3];
      for(unsigned int j = 0; j < 3; ++j){
        axis_v_index[j] = static_cast<int>
          (std::floor
           (cloud(i, j) * inverse_leaf_size) -
           static_cast<float>(min_b[j])
           );
      }

      // calc grid id
      int map_id = 0;
      for(unsigned int j = 0; j < 3; ++j)
        map_id += axis_v_index[j] * div_mul[j];

      leaves[map_id].points++;

      for(unsigned int j = 0; j < 3; ++j)
        leaves[map_id].mean[j] += cloud(i, j);
    }

    std::vector<int> erase_list;
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){
      //3点以上なら平均計算それ以下なら削除
      if(5 <= iter->second.points){
        for(unsigned int j = 0; j < 3; ++j){
          iter->second.mean[j] /= iter->second.points;
        }
      }
      else{
        erase_list.push_back(iter->first);
      }
    }

    // erase grid that less than 5 points.
    for(unsigned int erase_id = 0; erase_id < erase_list.size(); ++erase_id){
      auto erase_iter = leaves.find(erase_list[erase_id]);
      if( erase_iter != leaves.end()) leaves.erase(erase_iter);
    }

    // calc covariance
    for(unsigned int i = 0; i < cloud.rows(); ++i){
      int axis_v_index[3];
      for(unsigned int j = 0; j < 3; ++j)
        axis_v_index[j] = static_cast<int>
          (std::floor(cloud(i, j) * inverse_leaf_size) -
           static_cast<float>(min_b[j]));

      int map_id = 0;
      for(unsigned int j = 0; j < 3; ++j)
        map_id += axis_v_index[j] * div_mul[j];

      auto process_iter = leaves.find(map_id);
      if (process_iter != leaves.end()){

        for(unsigned int j = 0; j < 3; ++j){

          if(isinf(cloud(i, j) - leaves[map_id].mean[j])){
            return;
          }

          leaves[map_id].params[j] += pow(cloud(i, j) - leaves[map_id].mean[j], 2);
        }

        leaves[map_id].params[3] +=
          (cloud(i, 0) - leaves[map_id].mean[0]) *
          (cloud(i, 1) - leaves[map_id].mean[1]);

        leaves[map_id].params[4] +=
          (cloud(i, 0) - leaves[map_id].mean[0]) *
          (cloud(i, 2) - leaves[map_id].mean[2]);

        leaves[map_id].params[5] +=
          (cloud(i, 1) - leaves[map_id].mean[1]) *
          (cloud(i, 2) - leaves[map_id].mean[2]);

      }

    }

    // 逆行列計算
    erase_list.clear();
    // 3点以上なら平均計算それ以下なら削除
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){
      //cov作る
      CovarianceMatrixT cov;
      cov <<
        iter->second.params[0],
        iter->second.params[3],
        iter->second.params[4],
        iter->second.params[3],
        iter->second.params[1],
        iter->second.params[5],
        iter->second.params[4],
        iter->second.params[5],
        iter->second.params[2];

      //check付きinverse計算
      CovarianceMatrixT cov_inv;
      bool exists;
      double det = 0;
      cov.computeInverseAndDetWithCheck(cov_inv,det,exists);

      if(exists){//inverseがある場合→inverse_params[6]={0,0,0,0,0,0};11,22,33,12,13,23登録
        iter->second.inverse_params[0] = cov_inv(0,0);
        iter->second.inverse_params[1] = cov_inv(1,1);
        iter->second.inverse_params[2] = cov_inv(2,2);
        iter->second.inverse_params[3] = cov_inv(0,1);
        iter->second.inverse_params[4] = cov_inv(0,2);
        iter->second.inverse_params[5] = cov_inv(1,2);
      }
      else{
        erase_list.push_back(iter->first);
      }
    }

    for(unsigned int erase_id=0;erase_id<erase_list.size();erase_id++){//点削除
      auto erase_iter = leaves.find(erase_list[erase_id]);
      if( erase_iter != leaves.end()) leaves.erase(erase_iter);
    }

    int max_points=0;
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){
      if(max_points < iter->second.points){
        max_points = iter->second.points;
      }
    }

    //木を作る
	root_id=-1;
    nodes.resize(leaves.size());
	std::vector<std::vector<int>> axis_sort_ids(3,std::vector<int>(leaves.size()));
	point_with_id point_with_ids[leaves.size()];
    int point_count = 0;
    std::vector<int> index_map;
    index_map.resize(leaves.size());
	for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//voxel
        index_map[point_count] = iter->first;
		point_with_ids[point_count].id = point_count;
		point_with_ids[point_count].pos[0] = iter->second.mean[0];//mean
		point_with_ids[point_count].pos[1] = iter->second.mean[1];
		point_with_ids[point_count].pos[2] = iter->second.mean[2];
        point_count++;
	}
	for(sort_axis=0; sort_axis<3; sort_axis++){
		qsort(point_with_ids, leaves.size(), sizeof(point_with_id), AxisSort);
		for (unsigned int i=0 ; i < leaves.size() ; i++){
			axis_sort_ids[sort_axis][i]=point_with_ids[i].id;
		}
	}

	CreateNode
      (&root_id,
       leaves.size(),
       nodes,
       axis_sort_ids,
       0, // initial depth
       -1, // root id
       false);

    int root_map_id;
    root_map_id = index_map[root_id];
    root_id = root_map_id;

    for(unsigned int idx=0;idx<leaves.size();idx++){//voxel
        if(0<nodes[idx].parent_id) nodes_map[index_map[idx]].parent_id = index_map[nodes[idx].parent_id];
        else nodes_map[index_map[idx]].parent_id = -1;
        if(0<nodes[idx].left_id) nodes_map[index_map[idx]].left_id = index_map[nodes[idx].left_id];
        else nodes_map[index_map[idx]].left_id = -1;
        if(0<nodes[idx].right_id) nodes_map[index_map[idx]].right_id = index_map[nodes[idx].right_id];
        else nodes_map[index_map[idx]].right_id = -1;
        nodes_map[index_map[idx]].axis = nodes[idx].axis;
	}
  }


  TransformT Align(TransformT init_trans,
                   const Eigen::MatrixXd &cloud,
                   const unsigned int &max_iterations){
    unsigned int iterations = 0;
    double update_norm;
    bool converged = false;
    TransformT tf,update;
    tf = init_trans;
    ScoreParams gauss_params(output_prob, leaf_size);

    while(!converged){
      //このイテレーションのscore,jacobian,hessianの定義
      double iter_score = 0;
      JacobianT iter_jacobian = JacobianT::Zero();
      HessianT iter_hessian = HessianT::Zero();

      Eigen::MatrixXd tf_cloud = TransformPointCloud(cloud, tf);

      JacobianCoefficientsT jacobian_coefficients;
      HessianCoefficientsT hessian_coefficients;
      angleDerivatives(tf,jacobian_coefficients,hessian_coefficients);

      //点ループ
      for (unsigned int point_id = 0; point_id < tf_cloud.rows(); ++point_id){
        //点の型変換
        PointT point = tf_cloud.row(point_id);

        PointJacobianT point_jacobian;
        PointHessianT point_hessian;
        pointDerivatives
          (point,
           jacobian_coefficients,
           hessian_coefficients,
           point_jacobian,
           point_hessian);

        //近傍探索
        neighbor_list.clear();
        float target[3];
        for(int i = 0; i < 3; ++i)
          target[i] = tf_cloud(point_id, i);

        // rangeSearchRecursive(target, leaves, nodes_map, root_id, leaf_size);
        double temp = numeric_limits<double>::max();
        searchRecursive(target, leaves, nodes_map, root_id, temp);

        //近傍ループ
        // for (unsigned int neighbor_id = 0;neighbor_id < neighbor_list.size();neighbor_id++){
        // auto neighbor_iter = leaves.find(neighbor_list[neighbor_id]);
        auto neighbor_iter = leaves[neighbor_id];
          //分布の型変換
          PointT mean;
          mean <<
            neighbor_iter.mean[0],
            neighbor_iter.mean[1],
            neighbor_iter.mean[2];
          CovarianceMatrixT cov;
          cov <<
            neighbor_iter.params[0],
            neighbor_iter.params[3],
            neighbor_iter.params[4],
            neighbor_iter.params[3],
            neighbor_iter.params[1],
            neighbor_iter.params[5],
            neighbor_iter.params[4],
            neighbor_iter.params[5],
            neighbor_iter.params[2];

          CovarianceMatrixT cov_inv;
          bool exists;
          double det = 0;
          cov.computeInverseAndDetWithCheck(cov_inv,det,exists);

          if(!exists) continue;
          PointT x_k_dash = point - mean;
          //マッチ毎にscore,jacobian,hessianの計算

          tuple<double, JacobianT, HessianT> iter_derivatives = computeDerivative
            (gauss_params, point_jacobian, point_hessian, x_k_dash, cov_inv);
          iter_score += get<0>(iter_derivatives);
          iter_jacobian += get<1>(iter_derivatives);
          iter_hessian += get<2>(iter_derivatives);

#ifdef DEBUG
          jacobian_vector.push_back(get<1>(iter_derivatives));
          hessian_vector.push_back(get<2>(iter_derivatives));
#endif // DEBUG
          // } // neighbor
      } // points

      //ニュートン方第2項(update)の計算
      HessianT iter_hessian_inv = iter_hessian.completeOrthogonalDecomposition().pseudoInverse();;
      update = iter_hessian_inv * iter_jacobian;
      update_norm = update.norm();

#ifdef DEBUG
      jacobian_vector_sum.push_back(iter_jacobian);
      update_vector.push_back(update);
#endif // DEBUG

      //tfの再計算
      tf -= update;
      if(iterations >= max_iterations || update_norm < transformation_epsilon)
        converged = true;
      iterations++;
    }
    return tf;
  }

  void rangeSearchRecursive(const float* query_position,
                            const std::map <int, Leaf> &leaves,
                            const std::map <int, node> &tree,
                            const int &node_id,
                            double &search_range){
    // reach leave.
    if(node_id < 0){
      return;
    }

    node n = tree.at(node_id);
    Leaf l = leaves.at(node_id);

    double dist = EuclidDist3D(l.mean, query_position);

    if(dist < search_range){
      neighbor_list.push_back(node_id);
    }

    int next_id;
    int opp_id;
    if(query_position[n.axis] < l.mean[n.axis]){
      next_id = n.left_id;
      opp_id = n.right_id;
    }else{
      next_id = n.right_id;
      opp_id = n.left_id;
    }

    rangeSearchRecursive(query_position, leaves, tree, next_id, search_range);
    double diff = std::fabs(query_position[n.axis] - l.mean[n.axis]);
    if (diff < search_range)
      rangeSearchRecursive(query_position, leaves, tree, opp_id, search_range);
  }


#ifdef DEBUG
  std::vector<JacobianT> jacobian_vector;
  std::vector<JacobianT> jacobian_vector_sum;
  std::vector<TransformT> update_vector;
  std::vector<HessianT> hessian_vector;
#endif // DEBUG

  std::vector<int> neighbor_list;
  std::map<int, node> nodes_map;
  std::map<int, Leaf> leaves;
  std::vector<node> nodes;
};

#endif // __NDT_SCAN_MATCHER_HPP__
