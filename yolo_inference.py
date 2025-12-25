from ultralytics import YOLO

model = YOLO("C:\\Users\\prems\\Downloads\\Football-Analytics-with-Deep-Learning-and-Computer-Vision-master\\models\\best.pt")

results = model.predict(source="inputvideos/test.mp4", save=True)
print(results[0])
print("================================")
for box in results[0].boxes:
    print(box)
