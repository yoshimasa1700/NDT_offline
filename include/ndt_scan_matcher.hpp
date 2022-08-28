#ifndef __NDT_SCAN_MATCHER_HPP__
#define __NDT_SCAN_MATCHER_HPP__

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>

#define DEBUG

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

struct Node;
using NodePtr = shared_ptr<Node>;

class Leaf
{
public:
  vector<Vector3d> points;
  // Vector3f mean(0, 0, 0);
  Vector3d mean;
  float params[6] = {0,0,0,0,0,0};//11,22,33,12,13,23
  CovarianceMatrixT inverse_params;
  float a[9];
  float eigen_vector[9];
  // Matrix3f cov;
  // Matrix3f evecs;
  // Vector3f evals;
};

struct Node
{
  int idx;
  NodePtr leftChild;
  NodePtr rightChild;
  unsigned int axis;
};

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


double EuclidDist3D(const Vector3d &a, const float* b){
  float d2 = 0;
  for(int i = 0; i < 3; ++i){
    d2 += pow(a[i] - b[i], 2);
  }
  return sqrt(d2);
}


class KDTree{
  const unsigned int DIM = 3;

public:
  void build(vector<Leaf> &leaves){
    root_ = buildTreeRecursive(leaves.begin(),
                               leaves.size(),
                               0);
  }

  vector<int> rangeSearch(const float* query_position,
                   const vector<Leaf> &leaves,
                   const double &search_range){
    neighbor_list_.clear();

    rangeSearchRecursive(query_position,
                         leaves,
                         root_,
                         search_range);

    return neighbor_list_;
  }

private:
  NodePtr buildTreeRecursive(vector<Leaf>::iterator itr,
                             int point_count,
                             const int &depth)
  {
    if(point_count <= 0)
      return nullptr;

    const uint axis = depth % DIM;
    const int mid = point_count / 2;

    std::nth_element(itr,
                     itr + mid,
                     itr + point_count,
                     [&](Leaf &l,
                         Leaf &r)
                     {
                       return l.mean[axis] < r.mean[axis];
                     });

    NodePtr node = NodePtr(new Node());
    node->idx = mid;
    node->axis = axis;

    node->leftChild = buildTreeRecursive(itr, mid, depth + 1);
    node->rightChild = buildTreeRecursive
      (itr + mid + 1,
       point_count - mid,
       depth + 1);

    return node;
  }

  void rangeSearchRecursive(const float* query_position,
                            const vector<Leaf> &leaves,
                            const NodePtr &node_ptr,
                            const double &search_range){
    // reach leave.
    if(node_ptr == nullptr){
      return;
    }

    Leaf l = leaves.at(node_ptr->idx);

    double dist = EuclidDist3D(l.mean, query_position);

    if(dist < search_range){
      neighbor_list_.push_back(node_ptr->idx);
    }

    NodePtr next;
    NodePtr opp;

    bool need_search_opposit;

    if(query_position[node_ptr->axis] < l.mean[node_ptr->axis]){
      next = node_ptr->leftChild;
      opp = node_ptr->rightChild;

      need_search_opposit = l.mean[node_ptr->axis] < query_position[node_ptr->axis] + search_range;
    }else{
      next = node_ptr->rightChild;
      opp = node_ptr->leftChild;

      need_search_opposit = l.mean[node_ptr->axis] > query_position[node_ptr->axis] - search_range;
    }

    rangeSearchRecursive(query_position, leaves, next, search_range);

    if(need_search_opposit)
        rangeSearchRecursive(query_position, leaves, opp, search_range);
  }

  NodePtr root_;
  vector<int> neighbor_list_;
};


MatrixXd TransformPointCloud (const MatrixXd &cloud,
                              TransformT tf){
  Vector3d trans(tf(0),tf(1),tf(2));
  Matrix3d rot;
  rot = AngleAxisd(tf(3), Vector3d::UnitX())
    * AngleAxisd(tf(4), Vector3d::UnitY())
    * AngleAxisd(tf(5), Vector3d::UnitZ());

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
  vector<double> d_;
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
  static constexpr double transformation_epsilon = 0.01;
  double leaf_size = 0.08;
  double inverse_leaf_size_ = 1. / leaf_size;
  // double output_prob = 0.055;
  double output_prob = 0.0001;
  int neighbor_id;
  int root_id = -1;

  NDTCalc(){
  }


  array<int, 3> realToIndex(const array<double, 3> &point) const{

    array<int, 3> index;

    for(unsigned int i = 0; i < 3; ++i){
      index[i] = static_cast<int>
        (floor
         (point[i] * inverse_leaf_size_) - min_b[i]);
    }

    return index;
  }

  int calcMapId(const array<int, 3> &index) const{
    int map_id = 0;
    for(unsigned int j = 0; j < 3; ++j)
      map_id += axis_v_index[j] * div_mul[j];
    return map_id;
  }

  vector<Leaf> calcMean(vector<Leaf> &leaves){
    transform(
              leaves.begin(),
              leaves.end(),
              leaves.begin(),
              [](Leaf leaf){
                Leaf after(leaf);

                for(auto itr_p = leaf.points.begin();
                    itr_p != leaf.points.end();
                    ++itr_p){
                  after.mean += *itr_p;
                }

                after.mean /= leaf.points.size();

                return after;
              });

    return leaves;
  }


  vector<Leaf> calcCovariance(vector<Leaf> &leaves)
  {
    transform(
              leaves.begin(), leaves.end(),
              leaves.begin(),
              [](Leaf leaf){
                Leaf after(leaf);

                for(auto itr_p = leaf.points.begin();
                    itr_p != leaf.points.end(); ++itr_p){

                  Vector3d &p_ref = *itr_p;

                  for(unsigned int j = 0; j < 3; ++j){
                    after.params[j] +=\
                      pow(p_ref[j] - leaf.mean[j], 2);
                  }

                  after.params[3] +=
                    (p_ref[0] - leaf.mean[0]) *
                    (p_ref[1] - leaf.mean[1]);

                  after.params[4] +=
                    (p_ref[0] - leaf.mean[0]) *
                    (p_ref[2] - leaf.mean[2]);

                  after.params[5] +=
                    (p_ref[1] - leaf.mean[1]) *
                    (p_ref[2] - leaf.mean[2]);
                }

                for(unsigned int j = 0; j < 6; ++j){
                  after.params[j] /= leaf.points.size();
                }

                return after;
              });

    return leaves;
  }


  vector<Leaf> calcInvertMatrix(vector<Leaf> &leaves){
    // 3点以上なら平均計算それ以下なら削除
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){
      //cov作る
      CovarianceMatrixT cov;
      cov <<
        iter->params[0],
        iter->params[3],
        iter->params[4],
        iter->params[3],
        iter->params[1],
        iter->params[5],
        iter->params[4],
        iter->params[5],
        iter->params[2];

      //check付きinverse計算
      bool exists;
      double det = 0;
      cov.computeInverseAndDetWithCheck
        (iter->inverse_params,
         det, exists);

      if(!exists)
        leaves.erase(iter);
    }

    return leaves;
  }


  map<int, Leaf> createLeaves(const MatrixXd &cloud){
    // search min max point
    float min_p[3] = {numeric_limits<float>::max(),
                      numeric_limits<float>::max(),
                      numeric_limits<float>::max()};
    float max_p[3] = {0, 0, 0};

    for(unsigned int i = 0; i < cloud.rows(); ++i){
      for(int j = 0; j < 3; ++j){
        if(min_p[j] > cloud(i, j))
          min_p[j] = cloud(i, j);

        if(max_p[j] < cloud(i, j))
          max_p[j] = cloud(i, j);
      }
    }

    // calculation min max div voxel
    for(int i=0;i<3;i++){
      min_b[i] = static_cast<int>(floor(min_p[i] * inverse_leaf_size_));
      max_b[i] = static_cast<int>(floor(max_p[i] * inverse_leaf_size_));
      div_b[i] = max_b[i] - min_b[i] + 1;
    }

    div_mul[0] = 1;
    div_mul[1] = div_b[0];
    div_mul[2] = div_b[0] * div_b[1];

    map<int, Leaf> leaves;

    for(unsigned int i = 0; i < cloud.rows(); ++i){
      // grid quantization

      // calc grid id
      int map_id = calcMapId(realToIndex({cloud(i, 0),
                                          cloud(i, 1),
                                          cloud(i, 2)}));

      leaves[map_id].points.push_back(cloud.row(i));
    }

    return leaves;
  }

  void erase_if(vector<Leaf> &items,
                function<bool(const Leaf&)> cond) {
    for( auto it = items.begin(); it != items.end(); ) {
      if(cond(*it))
        it = items.erase(it);
      else
        ++it;
    }
  }

  void CreateMap(const MatrixXd &cloud){
    // NDT Map Generate
    map<int, Leaf> leaves = createLeaves(cloud);

    vector<Leaf> leaves_vec;

    for(auto itr = leaves.begin(); itr != leaves.end(); ++itr){
      leaves_vec.push_back(itr->second);
    }

    // filter leaf less than 5 points.
    erase_if(leaves_vec,
             [](const Leaf &leaf){
               return leaf.points.size() < 5;});

    // calc mean
    leaves_vec = calcMean(leaves_vec);

    // calc covariance
    leaves_vec = calcCovariance(leaves_vec);

    // 逆行列計算
    leaves_vec = calcInvertMatrix(leaves_vec);

    //木を作る
    kdtree_.build(leaves_vec);

    leaves_ = leaves_vec;
  }

  void searchRecursive(const float* query_position,
                       const vector<Leaf> &leaves,
                       const NodePtr &node_ptr,
                       double &min_dist){
    // reach leave.
    if(node_ptr == nullptr ){
      return;
    }

    // cout << " node_id = " << node_id;
    Leaf l = leaves[node_ptr->idx];
    double dist = EuclidDist3D(l.mean,
                               query_position);

    if(dist < min_dist){
      min_dist = dist;
      neighbor_id = node_ptr->idx;
    }

    NodePtr next;
    NodePtr opp;
    // cout << "left&right" << endl;
    if(query_position[node_ptr->axis] <
       l.mean[node_ptr->axis]){
      next = node_ptr->leftChild;
      opp = node_ptr->rightChild;
    }else{
      next = node_ptr->rightChild;
      opp = node_ptr->leftChild;
    }

    searchRecursive(query_position, leaves, next, min_dist);

    double diff = fabs(query_position[node_ptr->axis]
                       - l.mean[node_ptr->axis]);
    if (diff < min_dist)
      searchRecursive(query_position,
                      leaves, opp, min_dist);
  }

  TransformT Align(TransformT init_trans,
                   const MatrixXd &cloud,
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

      MatrixXd tf_cloud = TransformPointCloud(cloud, tf);

      JacobianCoefficientsT jacobian_coefficients;
      HessianCoefficientsT hessian_coefficients;
      angleDerivatives(tf,jacobian_coefficients,hessian_coefficients);

      unsigned int count = 0;
      unsigned int not_found_count = 0;

      //点ループ
      for (unsigned int point_id = 0; point_id < tf_cloud.rows(); ++point_id){
        PointT point = tf_cloud.row(point_id);
        PointT orig_point = cloud.row(point_id);

        PointJacobianT point_jacobian;
        PointHessianT point_hessian;
        pointDerivatives
          (orig_point,
           jacobian_coefficients,
           hessian_coefficients,
           point_jacobian,
           point_hessian);

        //近傍探索
        // neighbor_list.clear();
        float target[3];
        for(int i = 0; i < 3; ++i)
          target[i] = tf_cloud(point_id, i);

        vector<int> neighbor_list = kdtree_.rangeSearch(target, leaves_, leaf_size);
        // double temp = numeric_limits<double>::max();
        // searchRecursive(target, leaves, nodes_map, root_id, temp);

        if(neighbor_list.size() == 0){
          neighbor_not_found_points_.push_back(point);

          not_found_count++;

          continue;
        }

        neighbor_found_points_.push_back(point);

        //近傍ループ
        for (auto neighbor_itr = neighbor_list.begin() ;
             neighbor_itr != neighbor_list.end();
             ++neighbor_itr){

          Leaf l = leaves_.at(*neighbor_itr);

          //分布の型変換
          PointT mean;
          mean <<
            l.mean[0],
            l.mean[1],
            l.mean[2];
          CovarianceMatrixT cov;
          cov <<
            l.params[0],
            l.params[3],
            l.params[4],
            l.params[3],
            l.params[1],
            l.params[5],
            l.params[4],
            l.params[5],
            l.params[2];

          CovarianceMatrixT cov_inv;
          bool exists;
          double det = 0;
          cov.computeInverseAndDetWithCheck(cov_inv,det,exists);

          if(!exists){
            continue;
          }
          PointT x_k_dash = point - mean;

          if(x_k_dash.norm() > leaf_size){
            continue;
          }

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

          count++;
        } // neighbor
      } // points

      // std::cerr << "count: " << count << std::endl;
      // std::cerr << "not found count: " << not_found_count << std::endl;

      //ニュートン方第2項(update)の計算
      HessianT iter_hessian_inv = iter_hessian.completeOrthogonalDecomposition().pseudoInverse();;
      update = iter_hessian_inv * iter_jacobian;
      update_norm = update.norm();

      //tfの再計算
      tf -= update;

#ifdef DEBUG
      jacobian_vector_sum.push_back(iter_jacobian);
      hessian_vector_sum.push_back(iter_hessian);
      update_vector.push_back(tf);
      score_vector.push_back(iter_score);
#endif // DEBUG

      if(iterations >= max_iterations || update_norm < transformation_epsilon)
        converged = true;
      iterations++;
    }
    return tf;
  }

#ifdef DEBUG
  vector<JacobianT> jacobian_vector;
  vector<JacobianT> jacobian_vector_sum;
  vector<HessianT> hessian_vector_sum;
  vector<TransformT> update_vector;
  vector<double> score_vector;
  vector<HessianT> hessian_vector;

  vector<PointT> neighbor_found_points_;
  vector<PointT> neighbor_not_found_points_;
#endif // DEBUG

  vector<Leaf> leaves_;

  int axis_v_index[3];
  float div_mul[3];
  float min_b[3];
  float max_b[3];
  float div_b[3];

  KDTree kdtree_;
};

#endif // __NDT_SCAN_MATCHER_HPP__
