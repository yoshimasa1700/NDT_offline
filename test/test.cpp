#include "gtest/gtest.h"
#include <array>
#include <ndt_scan_matcher.hpp>

#include <random>
#include <vector>


using RndEng = std::default_random_engine;
using RndDist = std::uniform_real_distribution<float>;
using PointArrT = std::array<float, 3>;


void visualizeArr(const PointArrT &point){

  for(int i = 0; i < 3; ++i)
    cerr << point[i] << " ";
  cerr << endl;
}


void visualizeArr(const Vector3d &point){
  for(int i = 0; i < 3; ++i)
    cerr << point[i] << " ";
  cerr << endl;
}


// TEST(TestAxisSort, Simple1) {

//   point_with_id a, b;

//   unsigned int axis = 0;

//   a.pos[axis] = 2.0;
//   b.pos[axis] = 1.0;

//   EXPECT_EQ(AxisSort(axis, a, b), true);
// }


// TEST(TestCreateMap, Simple1) {
//   // create NDTCalc class.
//   NDTCalc ndt_calc;

//   // define leaf size.
//   ndt_calc.leaf_size = 2.0;

//   unsigned int sample_point_count = 20;

//   // create dummy point cloud.
//   MatrixXd result = MatrixXd::Zero(sample_point_count, 3);

//   for(unsigned int i = 0; i < sample_point_count; ++i){
//     result(i, 0) = 0.2 * i;
//     result(i, 1) = 0.0;
//     result(i, 2) = 0.0;
//   }

//   cerr << result << endl;

//   // create ndt map.
//   ndt_calc.CreateMap(result);

//   // check leaf count.
// }


Leaf InitLeaf(const vector<Vector3d> &points,
              const Vector3d &mean)
{
  Leaf l;
  l.points = points;
  l.mean = mean;

  return l;
}


PointArrT GenerateRandomPoint(RndEng &eng){

  const float MIN = -10.0;
  const float MAX = 10.0;
  RndDist distr(MIN, MAX);

  PointArrT point = {distr(eng),
                     distr(eng),
                     distr(eng)};

  return point;
}


void visualizeLeaf(const Leaf &l){

  for(int i = 0; i < 3; ++i){
    cerr << l.mean[i];
    if(i < 2)
      cerr << " ";
  }

  cerr << endl;
}


vector<Leaf> createRandomPoints(const unsigned int &gen_point_count){
  std::random_device rd;
  RndEng eng(rd());

  vector<Leaf> leaves(gen_point_count);
  for(unsigned int i = 0; i < gen_point_count; ++i){

    PointArrT point = GenerateRandomPoint(eng);
    Vector3d mean;

    mean << point[0], point[1], point[2];

    vector<Vector3d> points(10);
    leaves[i] = InitLeaf(points, mean);
  }

  return leaves;
}


TEST(TestCreateNode, Simple1) {
  // sample input.

  // create sample leaves
  unsigned int gen_point_count = 10;
  vector<Leaf> leaves = createRandomPoints(gen_point_count);

  for(uint i = 0; i < leaves.size(); ++i){
    visualizeLeaf(leaves[i]);
  }

  // build tree
  KDTree kd_tree_;

  cerr << "build kd tree" << endl;
  kd_tree_.build(leaves);

  // range search check

  // create ground truth in brute force.
  vector<int> gt_neighbor_list;

  // gen query position
  std::random_device rd;
  RndEng eng(rd());
  PointArrT query = GenerateRandomPoint(eng);

  // def range
  double search_range = 5.0;

  // get ground truth by brute force.
  for(unsigned int i = 0; i < leaves.size(); ++i){
    if(EuclidDist3D(leaves[i].mean , query.data()) < search_range){
      gt_neighbor_list.push_back(i);
    }
  }

  // NDTCalc ndt_calc;
  vector<int> neighbor_list = kd_tree_.rangeSearch(query.data(), leaves, search_range);

  // compare ground truth and neighbor_list
  visualizeArr(query);

  cerr << "gt neighbors" << endl;
  for(unsigned int i = 0 ; i < gt_neighbor_list.size(); ++i){
    visualizeArr(leaves[gt_neighbor_list[i]].mean);
  }

  ASSERT_EQ(neighbor_list.size(), gt_neighbor_list.size());
}
