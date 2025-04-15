import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# model loading 
model = torch.load("age_model.pt", map_location=torch.device('cpu'))  # Nếu chạy trên CPU
model.eval()

# initlize webcam
cap = cv2.VideoCapture(0)

# function prediction 
def predict_age(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # cũng tùy model
    ])
    input_tensor = transform(pil_img).unsqueeze(0)  # Thêm batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()  # nếu là regression

    return int(predicted_age)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Nếu nhấn 'p' → dự đoán độ tuổi và ghi lên ảnh
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        age = predict_age(frame)
        cv2.putText(display_frame, f"Age: {age}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Hiển thị webcam
    cv2.imshow("Age Estimation", display_frame)

    if key == ord('x'):
        break

# Giải phóng
cap.release()
cv2.destroyAllWindows()
