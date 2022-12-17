# line_detector
we built line detector model that can be used in line follower robot using camera instad of IR sensor

## Table of Contents

#### [prepare data](prepare_data)
 - preparing raw data for modeling (resizing, augementaion, labeling...)
 - finally : data are saved as numby array in files
 
#### [data](data)
 - the output of previous statge
 
#### [scripts sklearn](scripts_sklearn)
 - In this script we build line detector model using  Multilayer perceptron classifier in sklearn

#### [scripts pyspark](scripts_pyspark)
 - In this script we build line detector model using  Multilayer perceptron classifier in pyspark
