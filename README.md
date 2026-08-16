#  Fire & Smoke Object Detection

An end-to-end Computer Vision project for detecting **fire** and **smoke** in images using multiple deep learning object detection models.

The project includes dataset exploration, annotation analysis, model training and comparison, evaluation, and an interactive **Streamlit web application** for real-time inference.

---
## 🔗 Quick Links

-  [View Project Notebook](notebooks/Fire_Smoke_Object_Detection_Final.ipynb)
-  [Trained YOLOv11 Model](models/yolov11n_best.pt)
-  [Streamlit Application Code](app.py)
-  [Live Demo](https://fire-detection-project-nmwgog8wrwnmuejgsx9xzj.streamlit.app/)

##  Project Overview

Early detection of fire and smoke can help reduce damage and improve safety.

The objective of this project is to build an AI-based object detection system capable of identifying:

*  Fire
*  Smoke

The project explores and compares several modern object detection architectures and provides an interactive application for testing the trained model.

---

##  Project Objectives

* Understand and analyze the fire and smoke dataset.
* Perform professional Exploratory Data Analysis (EDA).
* Validate image and annotation quality.
* Analyze class distribution and bounding boxes.
* Train multiple object detection models.
* Compare model performance.
* Select a suitable model for deployment.
* Build an interactive Streamlit application.

---

##  Models Explored

The project includes experimentation with multiple object detection approaches:

* YOLOv8
* YOLOv11
* RF-DETR
* Faster R-CNN
* SSD MobileNet

These models were evaluated and compared to understand the trade-off between detection performance and computational efficiency.

---

##  Exploratory Data Analysis

A detailed EDA was performed before model development.

The analysis included:

* Dataset structure inspection
* Image-label matching
* Image quality validation
* Corrupted image detection
* Class distribution analysis
* Image size and format analysis
* Objects-per-image analysis
* Bounding box size analysis
* Bounding box position analysis
* Train / Validation / Test comparison
* Random visual annotation inspection
* Class-wise bounding box analysis
* Data quality assessment

---

##  Dataset

The dataset is organized in YOLO object detection format:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

### Classes

```text
0 → fire
1 → smoke
```

---

##  Key EDA Findings

* The dataset contains **8,456 images**.
* No corrupted images were detected.
* All images are RGB.
* All images are stored in JPEG format.
* All images have a resolution of **640 × 640 pixels**.
* Every image has a corresponding annotation file.
* The dataset contains **15,368 annotated objects**.
* Fire represents approximately **82%** of the annotations.
* Smoke represents approximately **18%**.
* The average number of objects per image is approximately **1.82**.
* Most images contain one or two objects.
* Many bounding boxes represent relatively small objects.
* Smoke bounding boxes tend to occupy larger regions than fire bounding boxes.
* Train, validation, and test sets have similar class distributions.

---

##  Main Dataset Challenge

The main issue identified during EDA is **class imbalance**.

Approximate class distribution:

```text
Fire  ≈ 82%
Smoke ≈ 18%
```

This imbalance should be considered when evaluating the trained model, especially when comparing performance for each class separately.

---

##  Streamlit Application

An interactive Streamlit application was developed to make the trained model easier to test.

### Application Features

* Upload an image
* Detect fire and smoke
* Display bounding boxes
* Display confidence scores
* Count detected fire regions
* Count detected smoke regions
* Adjust detection confidence
* Visualize detection results through a simple web interface

---

##  Repository Structure

```text
Fire-Smoke-Detection-Object-Detection/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── notebooks/
│   └── project_notebook.ipynb
│
└── models/
    └── trained_model.pt
```

> The notebook and trained model may be added separately depending on GitHub file-size limitations.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ayaewais8-ux/Fire-Smoke-Detection-Object-Detection.git
```

Move into the project folder:

```bash
cd Fire-Smoke-Detection-Object-Detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

##  Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

---

##  Technologies Used

* Python
* YOLO
* PyTorch
* Ultralytics
* OpenCV
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Pillow
* Streamlit
* Google Colab

---

##  AI Insights

The exploratory analysis showed that the dataset is clean and well structured for object detection.

The main modeling considerations are:

* Class imbalance between fire and smoke
* Presence of relatively small objects
* Different visual characteristics of fire and smoke
* Need to evaluate each class individually

These factors should be considered when interpreting model performance.

---

##  Future Improvements

Possible future improvements include:

* Collecting additional smoke samples
* Increasing environmental diversity
* Testing on more real-world images
* Improving small-object detection
* Evaluating performance on external datasets
* Deploying the application online
* Extending the system to video and live camera detection

---



**Aya Ewais**

Data Analysis | Machine Learning | Artificial Intelligence | Computer Vision

---

## ⭐🔥 Support

If you found this project useful or interesting, feel free to give the repository a ⭐.
