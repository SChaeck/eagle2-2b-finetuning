import json

# Read the original file and filter out task2 data (keep everything except task2)
filtered_data = []
with open('multitask_dataset.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        if data.get('task') != 'task2':  # Keep everything EXCEPT task2
            filtered_data.append(data)

print(f'Original dataset has {101} samples')
print(f'Filtered dataset (excluding task2) has {len(filtered_data)} samples')

# Write filtered data to new file
with open('no_task2_dataset_multitask.jsonl', 'w') as f:
    for data in filtered_data:
        f.write(json.dumps(data) + '\n')

print(f'Created no_task2_dataset_multitask.jsonl with {len(filtered_data)} samples')

# Show task distribution
task_counts = {}
for data in filtered_data:
    task = data.get('task', 'unknown')
    task_counts[task] = task_counts.get(task, 0) + 1

print('Task distribution in filtered dataset:')
for task, count in sorted(task_counts.items()):
    print(f'  {task}: {count} samples')