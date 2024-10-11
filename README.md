# PRJ400
# Food Image Segmentation using Detectron2 and FoodSeg103
This project implements a custom image segmentation pipeline using Detectron2, applied to the FoodSeg103 dataset. The goal is to accurately segment food items in images for potential applications in the food industry, such as meal recognition, dietary tracking, and automatic recipe assistance.

## ** Table of Contents **
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Configuration](#model-configuration)
6. [Results](#results)
7. [Challenges and Improvements](#challenges-and-improvements)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

# **Project Overview**
This repository showcases the implementation of an advanced image segmentation model using Detectron2. The FoodSeg103 dataset, which contains images of various food items, is used to train a model that can classify and segment different foods within an image.

The core of this project includes:

Preprocessing and handling the FoodSeg103 dataset.
Using Detectron2 to build, train, and evaluate a segmentation model.
Visualizing results and improving model performance through tuning.

# Dataset
The FoodSeg103 dataset consists of:

Images: JPEG images of various food items.
Segmentation Masks: Each food item has corresponding masks for segmentation.
Annotations: Color-coded annotations representing different food categories.
You can download the dataset from FoodSeg103 Dataset.

## Dataset Preprocessing
Image resizing and augmentation techniques were applied for better model performance.
RGB masks were converted to class indices using custom color-to-class mapping.
# Installation
To use this repository, you’ll need to install the required dependencies.

Clone the repository:

```bash
git clone https://github.com/craiglawsonnn/PR400.git
cd PR400
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Install Detectron2:
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
# Usage
After installing the required packages, you can run the model on your local machine.

Prepare the dataset: Ensure the dataset is placed in the appropriate directory:

```bash
/content/FoodSeg103/VOCdevkit/VOC/JPEGImages/
/content/FoodSeg103/VOCdevkit/VOC/Annotations/
```
Train the model: To start training the model with Detectron2:
```bash
python train.py --config-file configs/foodseg103.yaml --num-gpus 1
```
Evaluate the model: After training, you can evaluate the model:
```bash
python evaluate.py --config-file configs/foodseg103.yaml --num-gpus 1
```
# Model Configuration
The model uses a pre-trained backbone from the Detectron2 Model Zoo, which was fine-tuned for the FoodSeg103 dataset. Some notable configurations include:
- Learning Rate: 0.00025
- Batch Size: 16
- Optimizer: Adam
- Data Augmentation: Random crop, horizontal flip, and scaling

These configurations can be adjusted in the configs/foodseg103.yaml file.

## Results
#### Precision: Achieved a precision score of 87% on the test set.
#### Recall: The recall score was 85%, showing that the model captures most relevant segments.
#### IoU: Intersection over Union (IoU) metric reached 82%, indicating effective segmentation.
Here are some example results:

Input Image	Ground Truth	Predicted Mask
## Challenges and Improvements
#### Memory Management: Handling large datasets in memory was a challenge. This was mitigated using efficient data loading techniques and augmentation pipelines.
#### Model Tuning: Fine-tuning model hyperparameters such as learning rate and batch size significantly improved performance.
## Future Work
#### Multi-Label Classification: Extend the current single-label segmentation to multi-label classification, where multiple food items can be recognized in a single image.
#### Real-Time Inference: Optimize the model for real-time food segmentation on mobile and web platforms.
#### Dataset Expansion: Incorporate additional food datasets to improve the model's ability to generalize across different cuisines.

## Contributing
If you wish to contribute, please feel free to open a pull request or an issue. Contributions are always welcome!

# License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
