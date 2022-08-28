#include "gtest/gtest.h"
#include <array>
#include <ndt_scan_matcher.hpp>

#include <random>
#include <vector>


using RndEng = std::default_random_engine;
using RndDist = std::uniform_real_distribution<float>;
using PointArrT = std::array<float, 3>;


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
  unsigned int gen_point_count = 1000;
  vector<Leaf> leaves = createRandomPoints(gen_point_count);

  for(uint i = 0; i < leaves.size(); ++i){
    visualizeLeaf(leaves[i]);
  }

  // run
  // CreateNode
  //   (&root_id,
  //    leaves.size(),
  //    nodes,
  //    axis_sort_ids,
  //    depth,
  //    parent_id,
  //    node_is_right);

  // EXPECT_EQ(nodes[1].parent_id, -1);
  // EXPECT_EQ(nodes[1].left_id, 0);
  // EXPECT_EQ(nodes[1].right_id, 2);

  // range search check

  // vector<int> gt_neighbor_list;

  // // gen query position
  // PointArrT query = GenerateRandomPoint(eng);

  // // def range
  // double search_range = 5.0;

  // // get ground truth
  // for(unsigned int i = 0; i < points.size(); ++i){

  //   if(EuclidDist3D(points[i].data(), query.data()) < search_range){
  //     gt_neighbor_list.push_back(i);
  //   }
  // }

  // root_id = index_map[root_id];

  // map<int, node> nodes_map;

  // for(unsigned int idx = 0; idx < leaves.size(); idx++){//voxel
  //   if(0 <= nodes[idx].parent_id){
  //     nodes_map[index_map[idx]].parent_id = index_map[nodes[idx].parent_id];
  //   }else{
  //     nodes_map[index_map[idx]].parent_id = -1;
  //   }

  //   if(0 <= nodes[idx].left_id)
  //     nodes_map[index_map[idx]].left_id = index_map[nodes[idx].left_id];
  //   else
  //     nodes_map[index_map[idx]].left_id = -1;

  //   if(0 <= nodes[idx].right_id)
  //     nodes_map[index_map[idx]].right_id = index_map[nodes[idx].right_id];
  //   else
  //     nodes_map[index_map[idx]].right_id = -1;

  //   nodes_map[index_map[idx]].axis = nodes[idx].axis;
  // }

  // // range search
  // NDTCalc ndt_calc;

  // cerr << "root: " << root_id << endl;

  // ndt_calc.rangeSearchRecursive(query.data(), leaves, nodes_map, root_id, search_range);

  // // compare ground truth and neighbor_list

  // cerr << "query: " << endl;
  // visualizeArr(query);

  // cerr << "gt neighbors" << endl;
  // for(unsigned int i = 0 ; i < gt_neighbor_list.size(); ++i){
  //   visualizeArr(points[gt_neighbor_list[i]]);
  // }

  // ASSERT_EQ(ndt_calc.neighbor_list.size(), gt_neighbor_list.size());
}
