from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import requests  # <--- ADD THIS HERE

app = Flask(__name__)
CORS(app)

# 1. Load the Brain (Model and Vectorizer)
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Get text from the user
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'result': 'No text provided'})

    # 3. Predict using the ML Model
    transformed_text = tfidf_vectorizer.transform([text])
    ml_prediction = model.predict(transformed_text)[0]

    # 4. ALSO search the Live News API to see if it's reported on the internet!
    API_KEY = 'pub_0e7206ae0af54461b1706b1ff709704c'
    import urllib.parse
    import re
    # Clean the text (remove punctuation) and grab the first 4 words to make a flexible search
    clean_text = re.sub(r'[^\w\s]', '', text)
    query_words = " ".join(clean_text.split()[:4])
    search_query = urllib.parse.quote(query_words)
    
    # We search the general 'q' parameter which looks at titles and content
    url = f'https://newsdata.io/api/1/news?apikey={API_KEY}&q={search_query}&language=en'
    
    live_verified = False
    sources = []
    try:
        response = requests.get(url)
        api_data = response.json()
        # If the API found matching articles on the internet, we mark it as verified!
        if api_data.get('status') == 'success' and api_data.get('totalResults', 0) > 0:
            live_verified = True
            for article in api_data.get('results', [])[:3]: # grab top 3 sources
                sources.append({
                    'title': article.get('title', 'Unknown Title'),
                    'link': article.get('link', '#'),
                    'source_id': article.get('source_id', 'News Source')
                })
    except:
        pass

    # 5. Send the final answer back
    if live_verified:
        final_result = 'REAL (Verified active on Internet!)'
    else:
        # If not cleanly found live, we fallback to what the ML model thinks
        final_result = f'{ml_prediction} (Based on training data)'

    return jsonify({'result': final_result, 'sources': sources})

@app.route('/live-news', methods=['GET'])
def live_news():
    # Replace the text below with your real key from newsdata.io
    API_KEY = 'pub_0e7206ae0af54461b1706b1ff709704c' 
    url = f'https://newsdata.io/api/1/news?apikey={API_KEY}&country=in&language=en'
    
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get('results', [])
        
        live_predictions = []
        for article in articles[:5]: # We check the top 5 live headlines
            title = article.get('title', '')
            
            # Use your model to predict the live title
            vect_text = tfidf_vectorizer.transform([title])
            prediction = model.predict(vect_text)
            
            live_predictions.append({
                'title': title,
                'prediction': prediction[0], # Shows 'REAL' or 'FAKE'
                'link': article.get('link')
            })
        return jsonify(live_predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
