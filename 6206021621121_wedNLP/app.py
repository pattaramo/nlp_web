from multiprocessing import Value
from re import search
from unicodedata import name
from unittest import result
from flask import Flask, request, render_template, Markup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
from gensim.models.tfidfmodel import TfidfModel
from collections import Counter
import os
import spacy
from spacy import displacy
from spacy.tokens import Span
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

app = Flask(__name__,template_folder='template')

articles = []
article1 = []
article2 = []
ner = []
label = []

text = []

app.config["UPLOAD_PATH"] = "C:\\Users\\Meiji\\Downloads\\6206021621121_wedNLP-20221009T123232Z-001\\6206021621121_wedNLP\\text"
#home(request.files.getlist('Docfile'))


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def seve():
    '''textfile = request.files['textfile']
    textfile_path = textfile.filename
    textfile.save(textfile_path)'''
        
    if request.method == 'POST':
        i=1
        for text in request.files.getlist('textfile'):
            text.filename= f"{i}.txt"
            text.save(os.path.join(app.config['UPLOAD_PATH'],text.filename))
            i = i+1 
        #file= home(request.files.getlist('textfile'))     
    for i in range(5):    
        #if i != 0:
        #f = open(f"{i}.txt", "r")
        f = open(f".\\text\\{i+1}.txt", "r")
        #Read TXT file   
        article = f.read()
        # Tokenize the article: tokens
        tokens = word_tokenize(article)
        # Convert the tokens into lowercase: lower_tokens
        lower_tokens = [t.lower() for t in tokens]
        # Retain alphabetic words: alpha_only
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        # Remove all stop words: no_stops
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        # Instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        # Lemmatize all tokens into a new list: lemmatized
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        #list_article
        articles.append(lemmatized)
        #print(articles[0])
        #bow = Counter(lemmatized)
    
        #BOW
        #x = (bow.most_common(5))
        #article1.append(x)
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
    for word_id, word_count in sorted_word_count[:5]:
        article1.append(f"{dictionary.get(word_id)} {word_count}")
        
    
        #tf
        #if i != 0:
        #print (i)
        #n = int(i)
        #doc = corpus[i]
    doc = corpus[0]   
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1],reverse=True)
    for term_id, weight in sorted_tfidf_weights[:5]:
        y=(dictionary.get(term_id),weight)
        article2.append(y)
        #top_w.append(article1,article2)
      
        #ner spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(article)
    label=displacy.render(doc, style="ent")
    return render_template('index.html',article1=article1,article2=article2,label=Markup(label),dictionary=dictionary)


@app.route('/word',methods=['POST'])
def search():
    dictionary = Dictionary(articles)
    #request.method == 'GET'
    #word = User.query.filter_by(search=request.form['search']).first()
    word = request.form.get('ssearch')
    print(word)
    computer_id = (dictionary.token2id.get(word))
    print(computer_id)
    if computer_id is not None :
        havet = 'มีในเอกสาร'
        #print(computer_id)
    else:
        havet = 'ไม่มีในเอกสาร'
    return render_template('index.html',havet=havet,article1=article1,article2=article2,label=Markup(label))


model_path = "meiji"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = 512
def get_prediction(convert_to_label=False):
    news = request.form.get('news')
    # prepare our text into tokenized sequence
    inputs = tokenizer(news, padding=True, truncation=True, max_length= max_length, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

from sklearn.preprocessing import StandardScaler #แปลงข้อข้มูลมู (transform data)

@app.route('/news',methods=['POST'])
def Fakenews():
    news = request.form.get('news')
    new_df = news.replace
    sc = StandardScaler()
    newtran = sc.fit_transform(new_df)
    print(newtran)
    new_Label = newtran.apply(get_prediction)
    print(new_Label)
    return render_template('index.html',new_Label=new_Label)

if __name__ == "__main__":
    #app.run(host="0.0.0.0",port=80)
    app.run(debug=True)