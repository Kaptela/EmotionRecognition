import pandas as pd

train_labels_path = "train_labels.csv"
test_labels_path = "test_labels.csv"

train_labels = pd.read_csv(train_labels_path)
test_labels = pd.read_csv(test_labels_path)

train_labels['label'] = train_labels['label'] - 1
test_labels['label'] = test_labels['label'] - 1

train_labels.to_csv(train_labels_path, index=False)
test_labels.to_csv(test_labels_path, index=False)