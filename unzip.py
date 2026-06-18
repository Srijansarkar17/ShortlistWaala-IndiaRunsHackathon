import json

with open("/Users/srijansarkar/Documents/IndiaRunsHackathon/candidates.jsonl", "r") as f:
    candidates = [json.loads(line) for line in f if line.strip()]
print(len(candidates))  # 100000
