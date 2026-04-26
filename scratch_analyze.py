import pandas as pd

try:
    df = pd.read_csv('news.csv')
    print("Columns:", df.columns.tolist())
    print("Sample True Title:", df[df['label'] == 'REAL']['title'].iloc[0] if 'title' in df.columns else "No title")
    print("Sample Fake Title:", df[df['label'] == 'FAKE']['title'].iloc[0] if 'title' in df.columns else "No title")
    print("Dataset Size:", len(df))
except Exception as e:
    print('Error:', e)
