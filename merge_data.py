import pandas as pd

# 1. Load the separated files
# Ensure Fake.csv and True.csv are in the same folder as this script
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# 2. Assign labels so the model knows which is which
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# 3. Combine them into one single dataset
# This stacks them on top of each other
df_final = pd.concat([df_fake, df_true]).reset_index(drop=True)

# 4. Save the combined data as 'news.csv'
df_final.to_csv('news.csv', index=False)

print("✅ Success: Files merged into 'news.csv'!")
