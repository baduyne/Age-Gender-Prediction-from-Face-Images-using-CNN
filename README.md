**Objective**: 
To predict a person's age using a CNN model that extracts facial features from images.

**Dataset**:
I will be using the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset, which contains labeled facial images for age prediction.

**Approach**:
To tackle this problem, I have two options:
- Using a Pretrained Model (ResNet18):
    - I plan to reuse the ResNet18 architecture (a residual network with 18 layers, introduced in 2015). I will modify the final fully connected layer to suit a regression task, enabling the model to predict continuous age values.
- Building a Custom CNN from Scratch:
    - Alternatively, I may design and train a custom CNN tailored specifically for this task, focusing on optimizing the architecture for age prediction.

Thank you for your interest in my project!
