# Gender_classifier
> A gender classifier with **94% accuracy** of **testing sets** has been trained with **6000 faces**
## Usage
>* **crawling_image.ipynb** : crawling images by HTTP request
>* **haarCascade_face_detection.ipynb** : implements face detection by different harrcascade classifier
>* **extract_and_save_face.ipynb** : detect faces in the image, then crop and save
>* **train_gender_classifier.ipynb** : implements CNN model and training
>* **vgg_pre_trained_model.ipynb** : implements VGG16 model with weights pre-trained on ImageNet, but not suggest to only a few classes
>* **data_geneterator.ipynb** : generate batches of tensor image data with real-time data augmentation
>* **rectangle_face_mark_gender.ipynb** : implements face detection, then add rectangle and mark gender to different faces in the image
>* **gender_classify_middle_hiar_man.h5** : training weights of this classifier
## Dependencies
>* Python 3.5+
>* Tensorflow 1.2+
>* Keras 2.0+
>* OpenCV 3.1+ 
>* numpy, Pandas, PIL, matplotlib, requests
>* Anaconda 4.3, CPU: i7-4790 3.60GHz, GPU: GeForce GTX750, CUDA 8.0, cuDNN 5.0

## Results
![Alt text](https://github.com/jocialiang/gender_classifier/blob/master/results.jpg "Prediction")
## Environment setup
> Running deep learning model with **GPU acceleration**
>* **Windows**
>1. Is your VGA CUDA-Enabled? https://developer.nvidia.com/cuda-gpus
>2. Install CUDA https://developer.nvidia.com/cuda-downloads
>3. Install cuDNN https://developer.nvidia.com/cudnn <br />
>>* add ./cudnn/cuda/bin/cudnn64_5.dll to $PATH
>4. Install Anaconda https://www.anaconda.com/download/
>5. Create tensorflow-gpu shell, install tensorflow, keras and OpenCV by the following scripts<br />
>>* cmd 
>>* conda create --name tensorflow-gpu python=3.5 anaconda 
>>* activate tensorflow-gpu 
>>* pip install tensorflow-gpu 
>>* pip install keras 
>>* conda install -c menpo opencv3 
>>* python
>>* import tensorflow, keras, cv2
>>* `tensorflow.__version__` (check version)
>>* `keras.__version__`
>>* `cv2.__version__` (check OpenCV version)
>>* `deactivate tensorflow-gpu` (leave shell)

>* **Linux(Ubuntu16.04)**
>1. `nvidia-smi` (check VGA spec.)
>2. `apt-get update` <br />
>   `apt-get upgrade`
>3. install cuda
>4. install cudnn
>5. install anaconda
>6. Create tensorflow-gpu shell. Install tensorflow, keras and OpenCV by the following scripts<br />
>>* conda create -n tensorflow-gpu pyton=3.5
>>* source activate tensorflow-gpu
>>* conda install anaconda
>>* conda install -c conda-forge tensorflow-gpu
>>* conda install --channel https://conda.anaconda.org/menpo opencv3
>>* python
>>* import tensorflow, keras, cv2
>>* `tensorflow.__version__` (check version)
>>* `keras.__version__`
>>* `cv2.__version__` (check OpenCV version)
>>* `source deactivate tensorflow-gpu` (leave shell)


