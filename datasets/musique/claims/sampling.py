import json
import random

split = 'train'
num_samples_per_hop = {
    2: 400,
    3: 300,
    4: 300
}

with open(f'{split}.json') as f:
    data = json.load(f)

sampled_total = []
for hop, num_samples in num_samples_per_hop.items():
    data_hop = [item for item in data if item["num_hops"] == hop]
    sampled = random.sample(data_hop, num_samples)
    sampled_total.extend(sampled)

for i, item in enumerate(sampled_total):
    item["index"] = i

unsampled = [item for item in data if item not in sampled_total]
for i, item in enumerate(unsampled):  
    item["index"] = i 


save_fname = f'{split}_sampled.json'
save_fname_unsampled = f'{split}_unsampled.json'

with open(save_fname, 'w') as f:
    json.dump(sampled_total, f, indent=4)
with open(save_fname_unsampled, 'w') as f:
    json.dump(unsampled, f, indent=4)