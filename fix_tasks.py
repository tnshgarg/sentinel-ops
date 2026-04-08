import json

file1 = "tasks/hard/task_022_shadow_deception.json"
file2 = "tasks/hard/task_023_long_occlusion.json"

with open(file1, "r") as f:
    t1 = json.load(f)

t1["title"] = "Shadow Deception / Decoy Threat"
t1["correct_camera"] = "cam-01"
t1["correct_risk_level"] = "critical"
t1["should_escalate"] = True
t1["anomaly_start_frame"] = 1
t1["optimal_steps"] = 7
t1["max_steps"] = 15
t1["tags"] = ["decoy", "shadow"]

for frame in t1["frames"]:
    if "notes" in frame:
        frame["description"] = frame.pop("notes")
    else:
        frame["description"] = "No description provided."

with open(file1, "w") as f:
    json.dump(t1, f, indent=2)

with open(file2, "r") as f:
    t2 = json.load(f)

t2["title"] = "Long Occlusion Subject Tracking"
t2["correct_camera"] = "cam-02"
t2["correct_risk_level"] = "dangerous"
t2["should_escalate"] = True
t2["anomaly_start_frame"] = 3
t2["optimal_steps"] = 7
t2["max_steps"] = 15
t2["tags"] = ["occlusion", "tracking"]

for frame in t2["frames"]:
    if "notes" in frame:
        frame["description"] = frame.pop("notes")
    else:
        frame["description"] = "No description provided."

with open(file2, "w") as f:
    json.dump(t2, f, indent=2)

print("Tasks fixed!")
