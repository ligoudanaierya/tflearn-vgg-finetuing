import os
from tensorflow.python import pywrap_tensorflow
 
# code for finall ckpt
# checkpoint_path = os.path.join('~/tensorflowTraining/ResNet/model', "model.ckpt")
  
# code for designated ckpt, change 3890 to your num
checkpoint_path ="vgg16_weights.tfl"
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
