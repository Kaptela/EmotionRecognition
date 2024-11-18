import os
import pandas as pd

disgust_path = "disgust"
fear_path = "fear"

class_mapping = {
    disgust_path: 2,
    fear_path: 1
}

data = []

for folder, label in class_mapping.items():
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                data.append({"image": file, "label": label})

df = pd.DataFrame(data)

df.to_csv("new_image_labels.csv", index=False)

print(df.head())