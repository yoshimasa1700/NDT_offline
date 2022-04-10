#include "gtest/gtest.h"
#include <ndt_scan_matcher.hpp>


TEST(TestAxisSort, Simple1) {

    point_with_id a, b;

    unsigned int axis = 0;

    a.pos[axis] = 2.0;
    b.pos[axis] = 1.0;

    EXPECT_EQ(AxisSort(axis, a, b), true);
}
