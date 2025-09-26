import json

# Load your dataset
with open("fitness_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process the dataset
for item in data["data"]:
    for paragraph in item["paragraphs"]:
        # Remove context
        if "context" in paragraph:
            del paragraph["context"]

        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                # Remove answer_start
                if "answer_start" in answer:
                    del answer["answer_start"]

# Save the cleaned dataset
with open("fitness_dataset_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Preprocessing complete. Clean dataset saved as fitness_dataset_clean.json")
