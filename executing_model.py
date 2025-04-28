import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import os
import torchvision.models as models
import torch.optim as optim
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import time  
from argparse import ArgumentParser 
# --------------------- Model ---------------------
class adapt_resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        old_model = resnet18(pretrained=True)

        self.features = nn.Sequential(
            old_model.conv1,
            old_model.bn1,
            old_model.relu,
            old_model.layer1,
            old_model.layer2,
            old_model.layer3,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.log_var_age = nn.Parameter(torch.tensor(0.0))
        self.log_var_gender = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(0.5)

        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(256, 1),
        )

    def get_total_loss(self, age_out, age_label, gender_out, gender_label, age_criterion, gender_criterion):
        age_loss = age_criterion(age_out, age_label)
        gender_loss = gender_criterion(gender_out, gender_label)
        total_loss = (
            torch.exp(-self.log_var_age) * age_loss + self.log_var_age +
            torch.exp(-self.log_var_gender) * gender_loss + self.log_var_gender
        )
        return total_loss

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        age_pred = self.age_head(x)
        gender_pred = self.gender_head(x)
        return age_pred, gender_pred

# --------------------- Load model ---------------------
model = adapt_resnet18()
path = "Models/age_gender_predicting_model.pth.tar"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# --------------------- Predict function ---------------------
def predict_age_gender(frame):
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    # Transform the image to tensor format
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_img).unsqueeze(0)
    
    cater = ["Male", "Female"]
    
    # Predict age and gender
    with torch.no_grad():
        age, gender = model(input_tensor)
        predicted_age = age.item()
        gender = int(torch.round(torch.sigmoid(gender)))
    
    return int(predicted_age), cater[gender]

# --------------------- Predicting with image ---------------------
def predict_from_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image file not found.")
        return

    # Read the image from file
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Unable to read image.")
        return

    # Predict age and gender
    age, gender = predict_age_gender(frame)
    
    # Output the prediction
    print(f"Age: {age}, Gender: {gender}")

# --------------------- Predicting with Webcam ---------------------
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open webcam.")
    else:
        print("✅ Webcam opened successfully.")

    # --------------------- Regular Loop ---------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict age and gender
        age, gender = predict_age_gender(frame)

        # Display prediction on frame
        cv2.putText(frame, f"Age: {age}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        cv2.imshow("Age and Gender Estimation", frame)

        # Sleep to reduce lag
        time.sleep(2)

        # Press "X" to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------- Argument Parsing ---------------------
def parse_arguments():
    parser = ArgumentParser(description="Age and Gender Prediction")
    parser.add_argument("--mode", "-m", type=int, choices=[0, 1], required=True, 
                        help="Enter 0 to use webcam or 1 to predict from image.")
    parser.add_argument("--image_path", "-i", type=str, help="Path to the image for prediction (only used when mode is 1).")
    return parser.parse_args()

# --------------------- Main Function ---------------------
def main():
    args = parse_arguments()
    
    if args.mode == 0:
        print("🎥 Using webcam for prediction.")
        run_webcam()
    elif args.mode == 1:
        if args.image_path is None:
            print("❌ You need to provide the image path for prediction.")
        else:
            print(f"📸 Predicting from image: {args.image_path}")
            predict_from_image(args.image_path)
    else:
        print("❌ Invalid parameter. Please choose 0 for webcam or 1 for image.")

# --------------------- Run the application ---------------------
if __name__ == "__main__":
    main()
