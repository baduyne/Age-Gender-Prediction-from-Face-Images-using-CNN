# 🎯 Objective  
To predict a person’s **age** using a deep learning model that extracts facial features from images.  
Additionally, we aim to simultaneously predict the **gender** of the person using **multi-task learning (MTL)** to leverage shared features.

---

# 🗂️ Dataset  
We use the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new), which contains over 20,000 facial images labeled with:
- **Age** (0–116)
- **Gender** (0: Male, 1: Female)
- **Ethnicity** (not used in this task)

---

# 🧠 Approach  

## 🏗️ Model Architecture
We use a **pretrained ResNet18** as the backbone:
- Replace the final fully connected layer with two heads:
  - **Regression head** for age prediction (1 neuron, no activation)
  - **Classification head** for gender prediction (1 neuron, passed through sigmoid in inference)

## 🔧 Multi-Task Learning (MTL)
We train the model on both tasks simultaneously:

| Task     | Type               | Loss Function         |
|----------|--------------------|------------------------|
| Age      | Regression          | Mean Squared Error (MSE) |
| Gender   | Binary Classification | BCEWithLogitsLoss      |

---

# ⚠️ Problem: Loss Scale Imbalance

Age and gender losses operate on **different numerical scales**:
- Age loss (MSE) ≈ 10 to 100+
- Gender loss (BCE) < 1

✅ Solution: Learnable Uncertainty Weighting (Kendall et al.) We apply the method from the paper:
- 📄 "[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"](https://arxiv.org/abs/1705.07115)
   - Access Paper > View PDF. This equation is at the bottom of page 5 in the report.

- 🧪 Idea
  Each task is associated with a learnable uncertainty parameter (variance σ²), and the total loss is:
$$
\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{age}} + \log \sigma_1 + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{gender}} + \log \sigma_2
$$

- Where:

- $ \mathcal{L}_{\text{age}} $ and $ \mathcal{L}_{\text{gender}} $ are the individual losses for the age and gender tasks.
- $ \sigma_1^2 $ and $ \sigma_2^2 $ are the uncertainties (variances) for the age and gender tasks respectively. These uncertainties are learned as log-variance parameters during training, denoted $ \log \sigma_1 $ and $ \log \sigma_2 $.

This approach allows the model to automatically balance the contributions of the tasks based on their uncertainties. If one task is more uncertain, the model will give it less weight in the final total loss.

Thank you for your interest.

Best regards,

Duy
