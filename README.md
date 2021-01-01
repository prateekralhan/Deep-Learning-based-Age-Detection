# Deep-Learning-based-Age-Detection
Age detection is mainly the process of automatically discerning the age of a person solely from their face's photgraph. :smile:
Age detection is generally implemented as a two-stage process:

* **Stage #1:** 
Detect faces in the input image/video stream
* **Stage #2:** 
Extract the face Region of Interest (ROI), and apply the age detector algorithm to predict the age of the person

For Stage #1, any face detector capable of producing bounding boxes for faces in an image can be used, including but not limited to HAAR cascades, HOG + Linear SVM, Single Shot Detectors (SSDs), etc.

Exactly which face detector *we* use depends on your project:

* Haar cascades will be very fast and capable of running in real-time on embedded devices — the problem is that they are less accurate and highly prone to false-positive detections
* HOG + Linear SVM models are more accurate than Haar cascades but are slower. They also aren’t as tolerant with occlusion (i.e., not all of the face visible) or viewpoint changes (i.e., different views of the face)
* Deep learning-based face detectors are the most robust and will give you the best accuracy, but require even more computational resources than both Haar cascades and HOG + Linear SVMs

Once our face detector has produced the bounding box coordinates of the face in the image/video stream, we can move on to Stage #2 — identifying the age of the person. Given the bounding box (x, y)-coordinates of the face, we first extract the face ROI, ignoring the rest of the image/frame. Doing so allows the age detector to focus solely on the person’s face and not any other irrelevant “noise” in the image.The face ROI is then passed through the model, yielding the actual age prediction.

## Our age detector deep learning model
The deep learning age detector model we are using here today was implemented and trained by Levi and Hassner in their 2015 CVPR publication, [Age and Gender Classification Using Convolutional Neural Networks.](https://talhassner.github.io/home/publication/2015_CVPR)Here, the authors propose a simplistic AlexNet-like architecture that learns a total of eight age brackets:
* 0-2
* 4-6
* 8-12
* 15-20
* 25-32
* 38-43
* 48-53
* 60-100

We can notice that these age brackets are noncontiguous — this done on purpose, as the [Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender), used to train the model, defines the age ranges as such 😕.
We’ll be using a pre-trained age detector model here. 

## Installation:
***pip install -r requirements.txt*** will install the necessary dependencies.

## Usage:
1. Download Caffe model for age classification from [here](https://drive.google.com/file/d/1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW/view) and place it in the **age_detector** folder.

### Age Detection for Static Images
1. Run the Command: 
```
python detect_age.py --image images/1.png --face face_detector --age age_detector
```

### Age detection in real-time video streams:
1. Run the Command: 
```
python detect_age_video.py --face face_detector --age age_detector
```

## Outputs:


