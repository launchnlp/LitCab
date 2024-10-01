from datasets import load_dataset

dataset = load_dataset('sciq')

for split in dataset.keys():
    with open(f"{split}.txt", "w") as f:
        for example in dataset[split]:
            f.write(example["question"] + "\n" + example['correct_answer'] + "\n\n")