# tflearn-vgg-finetuing
tflearn vgg-finetuing
===

Fine-Tinetuing the VGG16 net by tflearn 

[build_h5.py](https://github.com/ligoudanaierya/tflearn-vgg-finetuing/blob/master/build_h5.py)用来创建h5格式的数据集

[vgg16_ft.py](https://github.com/ligoudanaierya/tflearn-vgg-finetuing/blob/master/vgg16_ft.py)导入预训练好的模型的参数并训练自己的数据集

介于当前网络上流传教广的vgg16的模型参数是一份.npz文件

所以可以在[这里](https://github.com/kentsommer/VGG16-Image-Retrieval/releases/download/v1.0/vgg16_weights.npz)下载该文件并运行代码即可微调自己的数据
