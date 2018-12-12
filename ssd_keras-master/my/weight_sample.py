import h5py
import numpy as np
import shutil

from misc_utils.tensor_sampling_utils import sample_tensors


# TODO: Set the path for the source weights file you want to load.

weights_source_path = '/home/ogai1234/lala/ssd_keras-master/weights/VGG_coco_SSD_300x300_iter_400000.h5'

# TODO: Set the path and name for the destination weights file
#       that you want to create.

weights_destination_path = '/home/ogai1234/lala/ssd_keras-master/weights/VGG_coco_SSD_300x300_iter_400000_subsampled_16_classes.h5'

# Make a copy of the weights file.
shutil.copy(weights_source_path, weights_destination_path)

# Load both the source weights file and the copy we made.
# We will load the original weights file in read-only mode so that we can't mess up anything.
weights_source_file = h5py.File(weights_source_path, 'r')
weights_destination_file = h5py.File(weights_destination_path)

classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']

conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

n_classes_source = 81
#
# Here are the indices of the 9 classes in MS COCO that we are interested in:
#
# `[0, 1, 2, 3, 4, 6, 8, 10, 12]`
#
# The indices above represent the following classes in the MS COCO datasets:
#
# `['background', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']`
#
# How did I find out those indices? I just looked them up in the annotations of the MS COCO dataset.
#
# While these are the classes we want, we don't want them in this order. In our dataset, the classes happen to be in the following order as stated at the top of this notebook:
#
# `['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'traffic_light', 'motorcycle', 'bus', 'stop_sign']`
#
# For example, '`traffic_light`' is class ID 5 in our dataset but class ID 10 in the SSD300 MS COCO model. So the order in which I actually want to pick the 9 indices above is this:

# `[0, 3, 8, 1, 2, 10, 4, 6, 12]`

#classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
# 在coco数据集中,这些选中的类的号码是:
# [aeroplane:5, bicycle:2 ,bird:16 ,boat:9 ,bus:6 ,car:3 ,cat:17 ,cow:21 ,dog:18 ,horse:19 ,motorbike:4 ,person:1 ,sheep:20 ,train:7 ,umbrellaman:28 ,cone: 11 ]
# 一共16+1类
classes_of_interest = [0,5,2,16,9,6,3,17,21,18,19,4,1,20,7,28,11]

subsampling_indices = []
for i in range(int(324/n_classes_source)):
    indices = np.array(classes_of_interest) + i * n_classes_source
    subsampling_indices.append(indices)
subsampling_indices = list(np.concatenate(subsampling_indices))

print(subsampling_indices)

# TODO: Set the number of classes in the source weights file. Note that this number must include
#       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
n_classes_source = 81
# TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
#       In case you would like to just randomly sample a certain number of classes, you can just set
#       `classes_of_interest` to an integer instead of the list below. Either way, don't forget to
#       include the background class. That is, if you set an integer, and you want `n` positive classes,
#       then you must set `classes_of_interest = n + 1`.
classes_of_interest = [0,5,2,16,9,6,3,17,21,18,19,4,1,20,7,28,11]
# classes_of_interest = 9 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

for name in classifier_names:
    # Get the trained weights for this layer from the source HDF5 weights file.
    kernel = weights_source_file[name][name]['kernel:0'].value
    bias = weights_source_file[name][name]['bias:0'].value

    # Get the shape of the kernel. We're interested in sub-sampling
    # the last dimension, 'o'.
    height, width, in_channels, out_channels = kernel.shape

    # Compute the indices of the elements we want to sub-sample.
    # Keep in mind that each classification predictor layer predicts multiple
    # bounding boxes for every spatial location, so we want to sub-sample
    # the relevant classes for each of these boxes.
    if isinstance(classes_of_interest, (list, tuple)):
        subsampling_indices = []
        for i in range(int(out_channels / n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
    elif isinstance(classes_of_interest, int):
        subsampling_indices = int(classes_of_interest * (out_channels / n_classes_source))
    else:
        raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")

    # Sub-sample the kernel and bias.
    # The `sample_tensors()` function used below provides extensive
    # documentation, so don't hesitate to read it if you want to know
    # what exactly is going on here.
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]],
                                          # The one bias dimension corresponds to the last kernel dimension.
                                          init=['gaussian', 'zeros'],
                                          mean=0.0,
                                          stddev=0.005)

    # Delete the old weights from the destination file.
    del weights_destination_file[name][name]['kernel:0']
    del weights_destination_file[name][name]['bias:0']
    # Create new datasets for the sub-sampled weights.
    weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

# Make sure all data is written to our output file before this sub-routine exits.
weights_destination_file.flush()

conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

