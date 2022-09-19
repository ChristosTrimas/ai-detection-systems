# Thesis Title

"Comparison of Artificial Intelligence Systems for the detection of objects on UAV-based images".

Unmanned Aerial Vehicles (UAVs) have experienced great growth and as of 2020 at
least 100 countries use UAVs in tactical missions, while at the same time even more
commercial applications deploy drones, for example photography and filmmaking,
smart crops, smart cities, emergency handling, drug delivery, traffic management, etc.
The big success of UAVs came due to the huge growth of electronics and the revolution
of data. One of the most popular application of drones is object detection before
designing the planned operation, e.g. differentiate pedestrians from cars or bikes in
cross-road management systems. Deep Learning algorithms have been proven to be the
best solution in such kind of problems. This diploma thesis collects and studies some
of the most well-known detection systems, it analyzes in theory and in practice an object
detector, the famous single-stage detector RetinaNet. Furthermore, a modified model is
proposed that utilizes more Convolutional Blocks and combines features from different
levels of the Neural Network. The extra convolution block is a mirror of the Feature
Pyramid Network; therefore, the new model is called “Two-Phase Feature Pyramid
Network Retina”. Since the goal is to compare those models, the classic RetinaNet and
the modified model, were trained and tested using the Stanford Drone Dataset, a dataset
designed to train object detectors for UAVs. The modified model achieves an accuracy
score 6% higher than the baseline model, and it seems to outperform the original model
in every metric, such as Precision, Sensitivity and F1 score. Finally, both the original
and the modified Retina, were compared with other well-known object detectors such
as YOLO, Faster RCNN, SSD, etc. The proposed architecture seems to outperform
almost every object detector from the literature in terms of mean Average Precision. In
conclusion, the modified model can be used to detect small objects in applications
where accuracy is a critical factor.
