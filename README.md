
---

## Multi-Task Learning (MTL)

We train the model to predict both age and gender **simultaneously**, using the following loss functions:

| Task     | Output Type         | Loss Function           |
|----------|---------------------|--------------------------|
| Age      | Regression           | `MSELoss` (Mean Squared Error) |
| Gender   | Binary Classification| `BCEWithLogitsLoss`     |

---

##  Challenge: Loss Scale Imbalance

The **age loss** (MSE) often has a larger magnitude than the **gender loss** (BCE), which can make training unstable.

###  Solution: Learnable Uncertainty Weighting

We apply the technique from:
> *"[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)"* (Kendall et al.)

### Idea:
The model learns task-specific uncertainty parameters ($\sigma^2$) that automatically scale each loss:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{total}}=\frac{1}{2\sigma_1^2}\mathcal{L}_{\text{age}}&plus;\log\sigma_1&plus;\frac{1}{2\sigma_2^2}\mathcal{L}_{\text{gender}}&plus;\log\sigma_2" title="Multi-task loss formula"/>
</div>

Where:
- L_age = Age regression loss (MSE)
- L_gender = Gender classification loss (BCE)
- σ₁, σ₂ = Learned uncertainties (as log-variance)

This allows the model to **dynamically balance** the contributions of each task based on its uncertainty during training.

---

##  Run the Project

```bash
# Clone the repository
git clone https://github.com/baduyne/Age-Gender-Prediction-from-Face-Images-using-CNN.git
cd Age-Gender-Prediction-from-Face-Images-using-CNN
```
 How to Run from Command Line
You can run the model in two modes:
-  Mode 0: Use Webcam:
```bash
python executing_model.py --mode 0
# or shorthand
python executing_model.py -m 0
```
-  Mode 1: Predict from a Static Image
```bash
python executing_model.py --mode 1 --image_path path_to_image.jpg
# or shorthand
python executing_model.py -m 1 -i path_to_image.jpg
```
 Requirements
Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```
(づ｡◕‿‿◕｡)づ  Thank you for your interest on my project.
