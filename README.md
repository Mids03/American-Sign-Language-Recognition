# American Sign Language Recognition

Real-time American Sign Language (ASL) recognition using a Convolutional Neural Network (CNN). This project processes live video feed from a webcam and predicts the corresponding ASL alphabet based on hand gestures.

## Repository Overview

This repository includes the following:

- **`preprocessing_and_training.ipynb`**: Handles data preprocessing and training a CNN model for ASL recognition.
- **`real_time.ipynb`**: Implements real-time recognition using the trained model and a webcam feed.
- **`sign_language`**: The trained model obtained from running the `preprocessing_and_training.ipynb` on a GPU.

## Requirements

To run the notebooks and utilize the functionalities, you need the following:

- **Python 3**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **Matplotlib**
- **CUDA 9.0** (for GPU acceleration)

Install the required packages via pip:

```bash
pip install tensorflow keras opencv-python matplotlib
```

## Dataset
This project uses the [Kaggle American Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist), which consists of 24 classes representing ASL alphabets (excluding J and Z due to motion requirements).

## Functionality
### Preprocessing and Training
The `preprocessing_and_training.ipynb` notebook performs the following:

1. **Preprocessing**: Loads the dataset, normalizes image pixel values, and encodes labels.
2. **Model Training**: Trains a CNN with the following architecture:
```
CONV2D -> RELU -> MAXPOOLING -> CONV2D -> RELU -> MAXPOOLING -> DROPOUT -> CONV2D -> RELU -> MAXPOOLING -> DROPOUT -> FLATTEN -> DENSE -> DROPOUT ->  DENSE -> SOFTMAX
```
3. Results:
    - Training Accuracy: 99.64%
    - Test Accuracy: 97.02%
The trained model is saved for use in real-time recognition.

### Real-Time Prediction Using Webcam
The `real_time.ipynb` notebook implements real-time ASL recognition. The process:

1. A Region of Interest (ROI) is defined as a green box in the webcam feed.
2. The user places their hand inside the ROI and performs a gesture.
3. The model predicts the corresponding alphabet based on the gesture.

### Neural Network Details
- Model optimizes using a categorical cross-entropy loss and an Adam optimizer.
- Dropout layers are included to prevent overfitting.


## Usage Instructions
1. **Dataset Setup**: Download the dataset from this [Kaggle link](https://www.kaggle.com/datasets/datamunge/sign-language-mnist ) and place it in the appropriate directory.
2. **Model Training**:
Run preprocessing_and_training.ipynb to train the model or use the pre-trained model provided.
3. **Real-Time Prediction**:
    - Run `real_time.ipynb` and ensure a webcam is connected.
    - Place your hand in the green box and perform gestures for prediction.


## References
- Dataset: [Kaggle - Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- CUDA for GPU acceleration: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
