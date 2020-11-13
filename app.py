import pandas as pd
from flask import Flask, jsonify, request
import pickle

import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def remove_features(data_str):
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    topic_re = re.compile('#(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    emoji_re = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
        
    # convert to lowercase
    data_str = data_str.lower()
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    # remove @mentions
#     data_str = mention_re.sub(' ', data_str)
    # remove #topic
    data_str = topic_re.sub(' ', data_str)
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    # remove emoji
    data_str = emoji_re.sub(' ', data_str)
    # remove non a-z 0-9 characters and words shorter than 1 characters
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 1:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 1:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    return " ".join(cleaned_str.split())

def rm_stopwords_lemmatize(column,stop_words,lemmer):
    filtered_col = column.map(lambda x : ' '.join([w for w in x.split() if w not in stop_words])) \
    .map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split() if w not in stop_words])) \
    .map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
    
    return filtered_col




# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    text_list = data['text']
    text = ' '.join(text_list)
    df = pd.DataFrame({'text':text},index=[0])

    lemmer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    df['filter_text'] = df['text'].apply(lambda x: remove_features(x))
    df['filter_text'] = rm_stopwords_lemmatize(df['filter_text'],stop_words,lemmer)
    result = model.decision_function(df['filter_text'])
    score = ((result[0]+2.6)/4.7)*6+1

    # # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    # # predictions
    # result = model.predict(data_df)

    # send back to browser
    output = {'results': round(score,3)}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)