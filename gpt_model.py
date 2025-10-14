import pandas as pd
from datasets import Dataset, DatasetDict

train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_df,
    "test": test_dataset
})

print(dataset)
print(dataset["train"][0])
