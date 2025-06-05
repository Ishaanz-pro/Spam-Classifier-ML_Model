from flask import Flask, request, render_template
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    transformed_input = transform_text(user_input)
    vec_input = vectorizer.transform([transformed_input])
    prediction = model.predict(vec_input)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', user_input=user_input, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
