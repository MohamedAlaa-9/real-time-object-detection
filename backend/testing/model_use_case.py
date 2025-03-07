from ultralytics import YOLO
import cv2

model_path = "backend/models/yolo11n.pt"
model = YOLO(model_path)

image_path = "backend/models/NBX.png"
results = model(image_path)

for r in results:
    im_array = r.plot()
    cv2.imshow("YOLOv11 Detection", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
