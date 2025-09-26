import pandas as pd
import glob
import os

# Root directory where ALS and Control folders are located
out_root = "/Users/samriddhsingh/Documents/kushal-final-year-project/processed_dataset"

rows = []
for label in ["ALS", "Control"]:
    for f in glob.glob(os.path.join(out_root, label, "*.wav")):
        rows.append([f, label])

df = pd.DataFrame(rows, columns=["file", "label"])
# print(df.values)
df.to_csv("dataset_metadata.csv", index=False)

print("✅ Metadata CSV created!")
