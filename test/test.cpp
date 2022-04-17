#include "gtest/gtest.h"
#include <ndt_scan_matcher.hpp>


TEST(TestAxisSort, Simple1) {

  point_with_id a, b;

  unsigned int axis = 0;

  a.pos[axis] = 2.0;
  b.pos[axis] = 1.0;

  EXPECT_EQ(AxisSort(axis, a, b), true);
}


TEST(TestCreateMap, Simple1) {
  // create NDTCalc class.
  NDTCalc ndt_calc;

  // define leaf size.
  ndt_calc.leaf_size = 2.0;

  unsigned int sample_point_count = 20;

  // create dummy point cloud.
  MatrixXd result = MatrixXd::Zero(sample_point_count, 3);

  for(unsigned int i = 0; i < sample_point_count; ++i){
    result(i, 0) = 0.2 * i;
    result(i, 1) = 0.0;
    result(i, 2) = 0.0;
  }

  cerr << result << endl;

  // create ndt map.
  ndt_calc.CreateMap(result);

  // check leaf count.
}


Leaf InitLeaf(const int &points, float *mean)
{
  Leaf l;
  l.points = points;
  memcpy(mean, l.mean, sizeof(float)*3);

  return l;
}

node InitNode(const int &parent_id, const int &left_id, const int &right_id, const int &axis){
  node n;
  n.parent_id = parent_id;
  n.left_id = left_id;
  n.right_id = right_id;
  n.axis = axis;
  return n;
}


TEST(TestCreateNode, Simple1) {
  // sample input.

  int root_id = -1;

  map<int, Leaf> leaves;

  for(int i = 0; i < 3; ++i){
    float mean[3] = {float(i), 0, 0};
    leaves[i] = InitLeaf(10, mean);
  }

  vector<node> nodes(leaves.size());

  // sort byt axis.
  vector<vector<int>> axis_sort_ids(3, vector<int>(leaves.size()));
  vector<point_with_id> point_with_ids(leaves.size());
  int point_count = 0;
  vector<int> index_map;
  index_map.resize(leaves.size());
  for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//voxel
    index_map[point_count] = iter->first;
    point_with_ids[point_count].id = point_count;
    point_with_ids[point_count].pos[0] = iter->second.mean[0];//mean
    point_with_ids[point_count].pos[1] = iter->second.mean[1];
    point_with_ids[point_count].pos[2] = iter->second.mean[2];
    point_count++;
  }

  for(unsigned int sort_axis=0; sort_axis<3; sort_axis++){
    sort(point_with_ids.begin(), point_with_ids.end(),
         [&](const point_with_id &n1, const point_with_id &n2)
         {return AxisSort(sort_axis, n1, n2);});
    for (unsigned int i=0 ; i < leaves.size() ; i++){
      axis_sort_ids[sort_axis][i]=point_with_ids[i].id;
    }
  }

  int depth = 0;
  int parent_id = -1;
  bool node_is_right = true;

  // run
  CreateNode
    (&root_id,
     leaves.size(),
     nodes,
     axis_sort_ids,
     depth,
     parent_id,
     node_is_right);

  EXPECT_EQ(nodes[1].parent_id, -1);
  EXPECT_EQ(nodes[1].left_id, 0);
  EXPECT_EQ(nodes[1].right_id, 2);
}
