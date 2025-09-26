import os
import shutil

root = "archive"
out_root = "processed_dataset"

# Make output folders
os.makedirs(os.path.join(out_root, "ALS"), exist_ok=True)
os.makedirs(os.path.join(out_root, "Control"), exist_ok=True)

# Mapping: Dysarthric = ALS, Control = Control
category_map = {
    "F_Dys": "ALS",
    "M_Dys": "ALS",
    "F_Con": "Control",
    "M_Con": "Control"
}

for group in os.listdir(root):
    if group not in category_map:
        continue
    label = category_map[group]
    group_path = os.path.join(root, group)
    for session in os.listdir(group_path):
        session_path = os.path.join(group_path, session)
        if not os.path.isdir(session_path):
            continue
        for wav_file in os.listdir(session_path):
            if wav_file.endswith(".wav"):
                src = os.path.join(session_path, wav_file)
                dst = os.path.join(out_root, label, f"{group}_{wav_file}")
                shutil.copy(src, dst)

print("✅ Dataset organized!")
