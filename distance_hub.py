import requests,re,json
from nltk.corpus import stopwords,brown
from nltk import WordNetLemmatizer as LM
from nltk import pos_tag
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def clean_text(corpus, keyword):
    my_stopwords = ['canada', 'canadian', 'united', 'state', 'country', 'field',
                    'area', 'level', 'wiki', 'http', 'google', 'year']
    print('Cleaning text with NLTK')
    cleaned_corpus = []
    for doc in corpus:
        cor = doc['article']
        token = cor.split(' ')
        token = [tk for tk in token if 3 < len(tk) < 20 or tk == keyword]
        token = [w for (w, p) in pos_tag(token) if p[0] == 'N' or w == keyword]
        token = [tk.lower() for tk in token]
        token = [tk for tk in token if tk.isalpha()]
        token = [LM().lemmatize(tk) for tk in token]
        token = [tk for tk in token if tk not in stopwords.words('english') or tk == keyword]
        # token = [tk for tk in token if tk in brown.words()]
        token = [tk for tk in token if tk not in my_stopwords or tk == keyword]
        text = ' '.join(token)
        cleaned_corpus.append(text)
    return cleaned_corpus  # return cleaned articles in list


def get_aritcle(keyword):
  url = 'http://en.wikipedia.org/w/api.php'
  headers = {'User-Agent': 'wikipedia (https://github.com/goldsmith/Wikipedia/)'}
  params = {
          'list': 'search',
          'srlimit': 50,
          'srsearch': keyword,
          'action': 'query',
          'format': 'json'
      }

  contents1 = requests.get(url, params, headers=headers).content.decode('utf-8')  # get wiki article list
  text = re.findall(r'"search":(\[.*\])', contents1)[0]
  files = json.loads(text)  # load list as json
  for f in files: # unpack json
      pageid = f['pageid']
      title = f['title']
      wordcount = f['wordcount']

      params2 = {
          'action': 'query',
          'format': 'json',
          'prop': 'extracts',
          "explaintext": "",
          "pageids": pageid
      }

      content2 = requests.get(url, params2, headers=headers).content.decode('utf-8')  # get article content page
      text = re.findall(r'"extract":"(.*)"', content2)[0]
      if text:
          f['article']=text  # update list json, add article content
  return files 

result = get_aritcle('data')

texts = [t['article'] for t in result]
len(texts)

clean_ts = clean_text(result, 'data')

from nltk import ngrams, tokenize

token = tokenize.word_tokenize(' '.join(clean_ts))

ngm = ngrams(token, 2)

grams = list(set(token)) + [' '.join(list(n)) for n in list(ngm)]

import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(model_url)

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(),tf.tables_initializer()])
  embeding = session.run(embed(grams))

len(embeding) == len(grams)


jj = {}
for i, gram in enumerate(grams):
  jj[gram] = np.linalg.norm(embeding[i]-embeding[grams.index('data')])

jj['data']
jj = sorted(jj.items(), key=lambda x: x[1])

jj[1000]
