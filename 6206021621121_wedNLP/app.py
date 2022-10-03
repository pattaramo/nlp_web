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

app.config["UPLOAD_PATH"] = "C:\\Users\\Meiji\\Downloads\\6206021621121_wedNLP-20220830T090746Z-001\\6206021621121_wedNLP\\text"
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
        f = open(f"./text/{i+1}.txt", "r")
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
    return (articles,article)

@app.route('/',methods=['POST'])
def bow():
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
    return render_template('index.html',article1=article1,article2=article2,label=Markup(label))

@app.route('/',methods=['POST'])
def tf(): 
    #tf
    #if i != 0:
    #print (i)
    #n = int(i)
    #doc = corpus[i]
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    tfidf_weights = defaultdict(int)
    for term_id, weight in itertools.chain.from_iterable(corpus):
        tfidf_weights[term_id] += weight
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1],reverse=True)
    for term_id, weight in sorted_tfidf_weights[:5]:
        article2.append(f"{dictionary.get(term_id)} {weight}")
    #top_w.append(article1,article2)
    return render_template('index.html',article1=article1,article2=article2,label=Markup(label))

def ner():  
    #ner spacy
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(corpus)
    label=displacy.render(doc, style="ent")
    return render_template('index.html',article1=article1,article2=article2,label=Markup(label))


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
        meiji = 'มีในเอกสาร'
        #print(computer_id)
    else:
        meiji = 'ไม่มีในเอกสาร'
    return render_template('index.html',meiji=meiji,article1=article1,article2=article2)

if __name__ == "__main__":
    #app.run(host="0.0.0.0",port=80)
    app.run(debug=True)