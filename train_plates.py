from ultralytics import YOLO
import torch


# # Load a model
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

torch.cuda.empty_cache()
# # Train the model with 2 GPUs
results = model.train(data='./data.yaml', epochs=10, imgsz=640,batch=8, device = "cuda") 

model.save('yolov5_trained.pt')  # Save the trained model to a file


# A=724, 308 B= 1203, 335 C = 401, 767 D= 1847, 775