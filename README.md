# DotCounter

This is a very simple project, my first foray into Deep learning with Tensorflow. The idea is instead of using a CNN in order to classify images into discrete categories such as 'dog' and 'cat', a simple CNN is used in order to count the number of objects in an image, without giving explicit information about what we are counting. This is a simplified test in using the weak indirect annotation of the number of objects within an image in the end to end application of counting objects without the need to first teach the program how to identify an object, then how to localize the object, and finally count the discrete instances of the object.   

### Help!

I am a complete beginner with regards to all this so if any kind soul could make sense of my findings and find a structure of CNN which produces high accuracy, or can at least explain why the accuracy of most of my architectures are so bad then that would be greatly appreciated!

Some areas where this may help other people:
- Creating a kind of image pipeline for your own data set of JPGs
	- I had a lot of trouble trying to figure out how to bring my own image data in to train a model, so this example can hopefully help some of you trying to do the same thing.
	- Note: Only works with very small images/data set!

- Converting MNIST into directories and subdirectories of JPGs

### Prerequisites

My setup is as follows:
	- Ubuntu 16.04 (Dualboot on windows machine)
	- Cuda toolkit 7.5
	- cuDNN 5.1
	- Tensorflow-0.11.0rc0
	- Python 3.5
	(https://pythonprogramming.net/how-to-cuda-gpu-tensorflow-deep-learning-tutorial/)


### Getting Started
**Step 1:**

Pull Datasets from Github
	- Download the 2 data directories, image_data and mnist_image_data 
	- Create an empty folder named saved_nets.
	- place all three within your DotCounter project folder

OR

Generate your own dot images and convert the mnist data set to jpgs using:
**dot_image_data_creator.py**
	- Make sure that the dotcounterdir string is set to the filepath of the DotCounter project folder in your system. (line16)
	- Dependancies:
		(for generating the dot images)
		scipy
		openCV
		(for converting MNIST)
		tensorflow
		PIL

**Step 2:**

Use one of the train scripts to train up a model, see more details in Findings.

**Step 3:**

Using **dot_implement_model.py** to implement the trained model:
	- Make sure to edit the architecture of the model to be exactly the same as the model that trained the saved variables.
	- Drag and drop JPG file into terminal, make sure the file path is in single inverted commas, ''.

```
$ python3 dot_implement_model.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
Please Enter Filepath of image enclosed in single inverted commas:'/home/shao/Documents/DotCounter/mnist_image_data/TEST_IMAGES/9/9_125.jpg' 
/home/shao/Documents/DotCounter/mnist_image_data/TEST_IMAGES/9/9_125.jpg
(1, 28, 28)
(1, 28, 28, 1)
(1, 28, 28, 1)
(1, 784)
(1, 784)
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.36GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
[[ -105.686409    -128.93051147   -21.71763039   377.47741699
    130.65116882 -1156.88842773  -520.42962646  -321.64532471
   -219.52412415  1535.15686035]]
The Predicted Result is:  [9]
```

### Findings

Result of using dot_V1CNNtrain.py on MNIST Dataset:
- Can see the trend of the accuracy increasing as the program progresses through the epochs.
- High Test Set Accuracy.

```
$ python3 dot_V1CNNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.38GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 10 loss: 620920.737366
Accuracy: 0.890625
Epoch 1 completed out of 10 loss: 85979.9612589
Accuracy: 0.914062
Epoch 2 completed out of 10 loss: 39210.6074371
Accuracy: 0.921875
Epoch 3 completed out of 10 loss: 25048.7630684
Accuracy: 0.960938
Epoch 4 completed out of 10 loss: 17330.4629747
Accuracy: 0.960938
Epoch 5 completed out of 10 loss: 13128.7457089
Accuracy: 0.960938
Epoch 6 completed out of 10 loss: 9967.8315143
Accuracy: 0.90625
Epoch 7 completed out of 10 loss: 7595.65426921
Accuracy: 0.960938
Epoch 8 completed out of 10 loss: 6473.12961066
Accuracy: 0.984375
Epoch 9 completed out of 10 loss: 5169.99286504
Accuracy: 0.96875
Test Set Accuracy: 0.93
Test Set Accuracy: 0.94
Test Set Accuracy: 0.92
Test Set Accuracy: 0.95
Test Set Accuracy: 0.95
Test Set Accuracy: 0.95
Test Set Accuracy: 0.93
Test Set Accuracy: 0.98
Test Set Accuracy: 0.95
Test Set Accuracy: 0.97
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt
```
Result of using dot_V1CNNtrain.py on dot image Dataset:
- Low Test Set Accuracy.
- Loss quickly drops to around 1000 and plateaus 

```
$ python3 dot_V1CNNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.37GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 10 loss: 1392637.91898
Accuracy: 0.4375
Epoch 1 completed out of 10 loss: 44973.1249313
Accuracy: 0.109375
Epoch 2 completed out of 10 loss: 1637.13985848
Accuracy: 0.117188
Epoch 3 completed out of 10 loss: 1161.69971228
Accuracy: 0.078125
Epoch 4 completed out of 10 loss: 1064.01476765
Accuracy: 0.0625
Epoch 5 completed out of 10 loss: 1020.98945403
Accuracy: 0.0625
Epoch 6 completed out of 10 loss: 1002.18014455
Accuracy: 0.09375
Epoch 7 completed out of 10 loss: 993.368439674
Accuracy: 0.0703125
Epoch 8 completed out of 10 loss: 989.831201792
Accuracy: 0.0703125
Epoch 9 completed out of 10 loss: 988.343240499
Accuracy: 0.101562
Test Set Accuracy: 0.08
Test Set Accuracy: 0.09
Test Set Accuracy: 0.1
Test Set Accuracy: 0.08
Test Set Accuracy: 0.06
Test Set Accuracy: 0.13
Test Set Accuracy: 0.06
Test Set Accuracy: 0.06
Test Set Accuracy: 0.11
Test Set Accuracy: 0.15
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt
```
Result of using **dot_V2CNNtrain.py** on dot image Dataset:
+ Changed the batch_size from 128 to 900
- Low Test Set Accuracy.
- Loss quickly drops steadily decreased and does not plateau (compare with dot_V1CNNtrain.py) 

```
$ python3 dot_V2CNNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.37GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 10 loss: 872011.186035
Accuracy: 0.151111
Epoch 1 completed out of 10 loss: 339159.40625
Accuracy: 0.283333
Epoch 2 completed out of 10 loss: 138515.962891
Accuracy: 0.366667
Epoch 3 completed out of 10 loss: 62697.184021
Accuracy: 0.387778
Epoch 4 completed out of 10 loss: 32450.2889099
Accuracy: 0.371111
Epoch 5 completed out of 10 loss: 17555.1549835
Accuracy: 0.296667
Epoch 6 completed out of 10 loss: 7862.0344162
Accuracy: 0.183333
Epoch 7 completed out of 10 loss: 1660.06092358
Accuracy: 0.117778
Epoch 8 completed out of 10 loss: 509.634899616
Accuracy: 0.126667
Epoch 9 completed out of 10 loss: 320.27294755
Accuracy: 0.112222
Test Set Accuracy: 0.14
Test Set Accuracy: 0.15
Test Set Accuracy: 0.08
Test Set Accuracy: 0.12
Test Set Accuracy: 0.14
Test Set Accuracy: 0.13
Test Set Accuracy: 0.07
Test Set Accuracy: 0.14
Test Set Accuracy: 0.13
Test Set Accuracy: 0.09
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt
```

Result of using dot_V3NNtrain.py on dot image Dataset:
+ batch_size =128, hm_epochs = 10
+ Uses normal NN instead of CNN
- Test Set Accuracy of around 0.6-0.7.
- Steadily decreases and does not plateau, (More data needed?)

```
$ python3 dot_V3NNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.36GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 10 loss: 14413613.2871
Accuracy: 0.414062
Epoch 1 completed out of 10 loss: 5979385.14453
Accuracy: 0.460938
Epoch 2 completed out of 10 loss: 4220062.1084
Accuracy: 0.484375
Epoch 3 completed out of 10 loss: 3113137.64795
Accuracy: 0.445312
Epoch 4 completed out of 10 loss: 2600226.29102
Accuracy: 0.640625
Epoch 5 completed out of 10 loss: 2098467.71069
Accuracy: 0.609375
Epoch 6 completed out of 10 loss: 1811022.66846
Accuracy: 0.71875
Epoch 7 completed out of 10 loss: 1576017.42114
Accuracy: 0.6875
Epoch 8 completed out of 10 loss: 1267072.35486
Accuracy: 0.726562
Epoch 9 completed out of 10 loss: 1149590.31824
Accuracy: 0.578125
Test Set Accuracy: 0.61
Test Set Accuracy: 0.59
Test Set Accuracy: 0.64
Test Set Accuracy: 0.69
Test Set Accuracy: 0.64
Test Set Accuracy: 0.58
Test Set Accuracy: 0.71
Test Set Accuracy: 0.56
Test Set Accuracy: 0.56
Test Set Accuracy: 0.71
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt
```

MOST SUCCESSFUL ARCHITECTURE FOR DOT DATA (so far)
Result of using dot_V3NNtrain.py on dot image Dataset:
+ batch_size =50, hm_epochs = 30
+ Test Set Accuracy of around 0.75-0.8.
- Loss steadily decreases and does not plateau.
- Epoch Accuracy in later epochs approach 0.9 or hiher (overfitting? More data needed?)

```
$ python3 dot_V3NNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.36GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 30 loss: 27705614.6318
Accuracy: 0.42
Epoch 1 completed out of 30 loss: 10936456.3784
Accuracy: 0.62
Epoch 2 completed out of 30 loss: 6832692.57593
Accuracy: 0.58
Epoch 3 completed out of 30 loss: 5030498.78271
Accuracy: 0.7
Epoch 4 completed out of 30 loss: 3785269.05005
Accuracy: 0.6
Epoch 5 completed out of 30 loss: 3162237.31628
Accuracy: 0.72
Epoch 6 completed out of 30 loss: 2647365.21002
Accuracy: 0.68
Epoch 7 completed out of 30 loss: 2276841.22916
Accuracy: 0.86
Epoch 8 completed out of 30 loss: 1913367.38403
Accuracy: 0.74
Epoch 9 completed out of 30 loss: 1754268.42737
Accuracy: 0.78
Epoch 10 completed out of 30 loss: 1611848.70041
Accuracy: 0.7
Epoch 11 completed out of 30 loss: 1387733.65724
Accuracy: 0.82
Epoch 12 completed out of 30 loss: 1306932.31924
Accuracy: 0.8
Epoch 13 completed out of 30 loss: 1132044.82167
Accuracy: 0.9
Epoch 14 completed out of 30 loss: 1082755.95644
Accuracy: 0.88
Epoch 15 completed out of 30 loss: 950869.666618
Accuracy: 0.86
Epoch 16 completed out of 30 loss: 851286.462097
Accuracy: 0.8
Epoch 17 completed out of 30 loss: 835982.46682
Accuracy: 0.9
Epoch 18 completed out of 30 loss: 703263.563892
Accuracy: 0.86
Epoch 19 completed out of 30 loss: 621431.843565
Accuracy: 0.96
Epoch 20 completed out of 30 loss: 651602.781251
Accuracy: 0.9
Epoch 21 completed out of 30 loss: 550063.647974
Accuracy: 0.92
Epoch 22 completed out of 30 loss: 557234.259968
Accuracy: 0.96
Epoch 23 completed out of 30 loss: 492977.634806
Accuracy: 0.76
Epoch 24 completed out of 30 loss: 455026.254133
Accuracy: 0.88
Epoch 25 completed out of 30 loss: 487130.864736
Accuracy: 0.94
Epoch 26 completed out of 30 loss: 374567.549484
Accuracy: 0.98
Epoch 27 completed out of 30 loss: 343344.35269
Accuracy: 0.94
Epoch 28 completed out of 30 loss: 376590.666821
Accuracy: 0.82
Epoch 29 completed out of 30 loss: 327883.224564
Accuracy: 0.86
Test Set Accuracy: 0.77
Test Set Accuracy: 0.79
Test Set Accuracy: 0.81
Test Set Accuracy: 0.76
Test Set Accuracy: 0.76
Test Set Accuracy: 0.76
Test Set Accuracy: 0.74
Test Set Accuracy: 0.77
Test Set Accuracy: 0.77
Test Set Accuracy: 0.78
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt

```

Result of using dot_V4CNNtrain.py on dot image Dataset:
+ batch_size =128, hm_epochs = 10
+ Tried to layer the fully connected layers which was succesful in dot_V3NNtrain.py on top of the CNN
- Test Set Accuracy low.
- Epoch accuracy low throughout.
- Loss steadily decreases and does not plateau.

```
python3 dot_V4CNNtrain.py 
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:01:00.0
Total memory: 1.94GiB
Free memory: 1.37GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:01:00.0)
Epoch 0 completed out of 10 loss: 446356968.031
Accuracy: 0.4375
Epoch 1 completed out of 10 loss: 46523464.1504
Accuracy: 0.390625
Epoch 2 completed out of 10 loss: 3516087.44627
Accuracy: 0.140625
Epoch 3 completed out of 10 loss: 136353.013227
Accuracy: 0.0703125
Epoch 4 completed out of 10 loss: 44998.3958929
Accuracy: 0.125
Epoch 5 completed out of 10 loss: 22019.1013646
Accuracy: 0.078125
Epoch 6 completed out of 10 loss: 12364.7582457
Accuracy: 0.117188
Epoch 7 completed out of 10 loss: 6846.56160665
Accuracy: 0.09375
Epoch 8 completed out of 10 loss: 4745.47561908
Accuracy: 0.109375
Epoch 9 completed out of 10 loss: 3006.2393446
Accuracy: 0.140625
Test Set Accuracy: 0.08
Test Set Accuracy: 0.14
Test Set Accuracy: 0.06
Test Set Accuracy: 0.09
Test Set Accuracy: 0.11
Test Set Accuracy: 0.11
Test Set Accuracy: 0.07
Test Set Accuracy: 0.16
Test Set Accuracy: 0.08
Test Set Accuracy: 0.12
Neural Net Variables Saved to Path:  /home/shao/Documents/DotCounter/saved_nets/dots_net.ckpt
```

### Resources

This was inspired by this paper: Learning to count with deep object features by Santi Segu´ı, Oriol Pujol, and Jordi Vitria (https://arxiv.org/pdf/1505.08082.pdf)

The code was based off tutorials from Sentdex and Morvan, thanks!
Sentdex - https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
Morvan - https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg

My interest was also piqued by Siragology 
Siragology - https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details