import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text)
    text = text.strip("[]").replace("\\n", " ").replace("\n", " ")
    text = text.replace("'", "").replace('"', "")
    text = text.replace(" ,", ",").replace(" ?", "?").replace(" .", ".")
    text = text.replace(" !", "!")
    text = " ".join(text.split())
    return text

def extract_dialog_column(df):
    possible_columns = ["dialog", "chat", "text", "utterance", "conversation", "Situation"]
    for col in possible_columns:
        if col in df.columns:
            if df[col].apply(lambda x: isinstance(x, str) and "[" in x and "]" in x).any():
                return df[col].apply(clean_text)
            else:
                return df[col].astype(str).apply(clean_text)
    if "personas" in df.columns:
        if "utterances" in df.columns:
            combined = df["personas"].astype(str) + " " + df["utterances"].astype(str)
        else:
            combined = df["personas"].astype(str)
        return combined.apply(clean_text)
    return pd.Series(dtype=str)

# BlendedSkillTalk dataset
bst_train = pd.read_csv("./data/BlendedSkillTalk/train.csv") # train 
bst_val = pd.read_csv("./data/BlendedSkillTalk/validation.csv") # validation 
bst_test= pd.read_csv("./data/BlendedSkillTalk/test.csv") # test

#DailyDialog dataset
dd_train = pd.read_csv("./data/DailyDialog/train.csv") # train
dd_val = pd.read_csv("./data/DailyDialog/validation.csv") # validation
dd_test= pd.read_csv("./data/DailyDialog/test.csv") # test

# Empathetic Dialogues datatset
ed_df = pd.read_csv("./data/EmpatheticDialogues/dataset.csv")

# Persona Chat dataset
pc_df = pd.read_csv("./data/PersonaChat/dataset.csv")

# Topical Chat dataset
tc_df = pd.read_csv("./data/TopicalChat/dataset.csv")

datasets = {
    "BlendedSkillTalk": [bst_train, bst_val, bst_test],
    "DailyDialog": [dd_train, dd_val, dd_test],
    "EmpatheticDialogues": [ed_df],
    "PersonaChat": [pc_df],
    "TopicalChat": [tc_df]
}

all_dialogs = []

for name, dfs in datasets.items():
    for df in dfs:
        dialog_col = extract_dialog_column(df)
        dialog_col = dialog_col[dialog_col.str.len() > 5]  # eliminar entradas vacías
        df_clean = pd.DataFrame({"dialog": dialog_col})
        df_clean["source"] = name
        all_dialogs.append(df_clean)

merged_df = pd.concat(all_dialogs, ignore_index=True)
merged_df.drop_duplicates(subset=["dialog"], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

print(f"\nTotal dialogues combined: {len(merged_df)}")
print("\nDividing dataset into train / validation / test")

train_df, temp_df = train_test_split(merged_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

os.makedirs("./data/UnifiedDataset", exist_ok=True)

train_df.to_csv("./data/UnifiedDataset/train.csv", index=False)
val_df.to_csv("./data/UnifiedDataset/validation.csv", index=False)
test_df.to_csv("./data/UnifiedDataset/test.csv", index=False)

print("\nDataset Ready")
print(train_df.head())
