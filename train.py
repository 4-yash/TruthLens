import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Load the merged data
df = pd.read_csv('news.csv')

# 2. Upgrade the Dataset Features!
# Previously it only trained on 'text'. Now we merge 'title' and 'text' 
# so it becomes a master at analyzing both short headlines AND long paragraphs.
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

# 3. Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=7)

# 4. Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# 4. Fit and transform
tfidf_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U')) 
tfidf_test = tfidf_vectorizer.transform(x_test.values.astype('U'))

# 5. Train the Model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 6. Save the model and vectorizer to files
pickle.dump(pac, open('model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("✅ Success! 'model.pkl' and 'tfidf_vectorizer.pkl' have been created.")
