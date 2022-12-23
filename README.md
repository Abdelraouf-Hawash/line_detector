# line_detector
In this repository I will share the **source code** of building line detector model that can be used in line follower robot using camera instad of IR sensor

<img src="media/line_detector.gif" width="800"/>


## Table of Contents

#### [prepare data](prepare_data)
 - preparing raw data for modeling (resizing, augementaion, labeling...)
 - finally : data are saved as numby array in files
 
 - The accuracy of this model can be improved by using more data than was used then prepare it again.  And train the modle on it again.
 
#### [data](data)
 - the output of previous statge
 - 4563 sample
 
#### [scripts sklearn](scripts_sklearn)
 - In this script we build line detector model using  Multilayer perceptron classifier in sklearn

#### [scripts pyspark](scripts_pyspark)
 - In this script we build line detector model using  Multilayer perceptron classifier in pyspark
