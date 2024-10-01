import pandas as pd

file_path = "/../../TruthfulQA/TruthfulQA.csv"
df = pd.read_csv(file_path)

questions = df["Question"].tolist()
answers = df["Correct Answers"].tolist()

questions_test = questions[:420]
answers_test = answers[:420]
questions_train = questions[420:]
answers_train = answers[420:]

split = ["train", "test"]
for s in split:
    with open(f"{s}.txt", "w") as f:
        if s == "train":
            for i in range(len(questions_train)):
                f.write(questions_train[i] + "\n" + answers_train[i].split(';')[0] + "\n\n")
        else:
            for i in range(len(questions_test)):
                f.write(questions_test[i] + "\n" + answers_test[i] + "\n\n")