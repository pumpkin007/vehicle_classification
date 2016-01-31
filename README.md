# Vehicle Classification

This code is for vehicle classification using DNN features.

## Instruction

### Run demo

Before running the code:

- Download this project to your computer.

- Install [VLFeat](http://www.vlfeat.org/) for generate fisher vector.

- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is already installed in [reference](https://github.com/pumpkin007/vehicle_classification/tree/master/reference) folder. You may follow the [instructions](https://github.com/cjlin1/libsvm) if you want to compile on your computer.

- Run [run.m](https://github.com/pumpkin007/vehicle_classification/blob/master/script/run.m) in [script](https://github.com/pumpkin007/vehicle_classification/tree/master/script) folder, then wait for a while, you can get the result.

### Generate DNN feature by yourself

If you want to generate DNN feature by yourself, follow the following procedure:

- Download the [vehicle dataset](http://goo.gl/nPSMOu).

- Modify the code in [set_dir.m](https://github.com/pumpkin007/vehicle_classification/blob/master/script/set_dir.m) so that [dirs.vehicle] points to the vehicle dataset. You may also set a new [dirs.feature] to where you want to store the generated feature.

- Install [Caffe](http://caffe.berkeleyvision.org/installation.html) for feature extraction on Alexnet. Also you need to install Matcaffe.

- Download [caffemodel for Alexnet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) and save it to your corresponding caffe model directory. Also copy the [deploy_single.prototxt](https://github.com/pumpkin007/vehicle_classification/blob/master/need_for_caffe/deploy_single.prototxt) to the caffe model directory.

- Modify the corresponding code in [set_dir.m](https://github.com/pumpkin007/vehicle_classification/blob/master/script/set_dir.m), regarding the directory of your caffe root directory.

- Run [run.m](https://github.com/pumpkin007/vehicle_classification/blob/master/script/run.m) in [script](https://github.com/pumpkin007/vehicle_classification/tree/master/script) folder, then you can get the generated feature and accuracy result.