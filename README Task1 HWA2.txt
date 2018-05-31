HWA2 Task 1

To run code for part i) new CNN trained from scratch

The datasets must be downloaded and stored in the same directory. 
The datasets are: X.npy and Y.npy and available to download here: 
https://www.kaggle.com/simjeg/lymphoma-subtype-classification-fl-vs-cll/data

Run: split_data_function.py
then run convert_array_to_image.py
Finally run cnn.py

To run code for part ii) training a CNN on pre-trained model VGG16:

Download the pretrained weights'vgg16_weights_tf_dim_ordering_tf_kernels.hy' from 
https://www.kaggle.com/keras/vgg16/data

Run split_data_function.py
then run resize_images.py
Finally run VGG16.py
