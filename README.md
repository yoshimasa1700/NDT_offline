# NDT offline

- This repository is an environment for offline evaluation of Normal Distribution Transform (NDT).
- Qualitative evaluation through visualization is possible.


## how to use

### build

```shell
mkdir build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build .
make -C build -j
```

### run(toy problem)

```shell
python3 ndt_sample.py
```

### run(point cloud data)

```shell
python3 ndt_sample_open3d.py
```