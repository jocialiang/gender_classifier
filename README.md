# Gender_classifier
> Experiments with **Tensorflow**, **Keras** and **OpenCV**
## Code
>* **crawling_image.ipynb** : crawling images by http request
>* **haarCascade_face_detection.ipynb** : implements face detection by different harrcascade classifier
>* **extract_and_save_face.ipynb** : detect faces in image, then crop and save
>* **train_gender_classifier.ipynb** : implements CNN model and training
>* **vgg_pre_trained_model.ipynb** : implements VGG16 model with weights pre-trained on ImageNet, but not suggest to only a few classes
>* **data_geneterator.ipynb** : generate batches of tensor image data with real-time data augmentation
>* **rectangle_face_mark_gender.ipynb** : implements face detection, add rectangle and mark gender to different faces in image
## Environment setup
> Runing deep learning model with gpu acceleration
>* **Windows**
>1. Is your VGA CUDA-Enabled? https://developer.nvidia.com/cuda-gpus
>2. Install CUDA https://developer.nvidia.com/cuda-downloads
>3. Install cuDNN https://developer.nvidia.com/cudnn <br />
>>* add ./cudnn/cuda/bin/cudnn64_5.dll to $PATH
>4. Install Anaconda https://www.anaconda.com/download/
>5. Create tensorflow-gpu shell, install tensorflow and keras by the following scripts<br />
>>* cmd <br />
>>* conda create --name tensorflow-gpu python=3.5 anaconda <br />
>>* activate tensorflow-gpu <br />
>>* pip install tensorflow-gpu <br />
>>* pip install keras <br />


