# Cranfield assignment: Object detection on train tracks

The aim of this assignment is to analyses images of a railway crossing in order to dectect different evenments. The differents events are the following :

- Event 0 : there is no event;
- Event 1 : The railway track is not clear of road/pedestrian traffic
- Event 2 : a road vehicle is entering the railway line crossing
- Event 3 : a road vehicle is leaving the railway line crossing
- Event 4 : the level crossing safety barrier is deployed
- Event 5 : rail traffic is currently using the railway track

This codes relies heavily on a mix of image processing/svm in order to detect the objects.

As of now, the data used is not available and the detection accuracy have to be manually calculated after running the code.

The assignment is available in the pdfs folder.

## Run

This code uses Opencv3 to run.

```bash

cmake .
make

./build/bin/modelsCreator

./build/bin/imageDecision <path_to_images>
```
