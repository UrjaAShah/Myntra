#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import os
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import random

cust_id = ""
print("Here-1")
df = pd.read_csv("D:\\kaggle\\HM_Dataset\\articles.csv")
print("Here0")
tt = pd.read_csv("D:\\kaggle\\HM_Dataset\\transactions_train.csv")
print("Here1")
tt_reduced = tt.head(1788324) 
comb = pd.merge(df, tt_reduced, on='article_id')
print("Here2")
# comb[comb["customer_id"]=="05ed96931b707698bc94aa53766d44686ae5ccbbc99dfbda8694ae811c54f28f"]


def get_desc(user_input,cust_id):
    cust_comb= comb[comb["customer_id"]==cust_id]
    print("Here")
    import pandas as pd    
    import random
    def select_word_by_probability(word_series):
        if word_series.empty:
            return None  # Handle empty word_series
        total_count = word_series.sum()
        if total_count == 0:
            return None
        total_count = word_series.sum()
        probabilities = word_series / total_count
        selected_word = random.choices(word_series.index, probabilities)[0]
        return selected_word

    product_type_counts = cust_comb['product_type_name'].value_counts().sort_values(ascending=False)
    graphical_appearance_counts = cust_comb['graphical_appearance_name'].value_counts().sort_values(ascending=False)
    perceived_colour_value_counts = cust_comb['perceived_colour_value_name'].value_counts().sort_values(ascending=False)
    perceived_colour_master_counts = cust_comb['perceived_colour_master_name'].value_counts().sort_values(ascending=False)
    index_counts = cust_comb['index_name'].value_counts().sort_values(ascending=False)
    garment_group_counts = cust_comb['garment_group_name'].value_counts().sort_values(ascending=False)
    
    def select_words_by_probability(word_series, num_words=5):
        if word_series.empty:
            return None  # Handle empty word_series
        total_count = word_series.sum()
        if total_count == 0:
            return None
        total_count = word_series.sum()
        probabilities = word_series / total_count
        selected_words = random.choices(word_series.index, probabilities, k=num_words)
        return selected_words


    product_type_counts = cust_comb['product_type_name'].value_counts().sort_values(ascending=False)
    graphical_appearance_counts = cust_comb['graphical_appearance_name'].value_counts().sort_values(ascending=False)
    perceived_colour_value_counts = cust_comb['perceived_colour_value_name'].value_counts().sort_values(ascending=False)
    perceived_colour_master_counts = cust_comb['perceived_colour_master_name'].value_counts().sort_values(ascending=False)
    index_counts = cust_comb['index_name'].value_counts().sort_values(ascending=False)
    garment_group_counts = cust_comb['garment_group_name'].value_counts().sort_values(ascending=False)


    
    input1 = select_word_by_probability(product_type_counts)
    input2 = select_word_by_probability(graphical_appearance_counts)
    input3 = select_word_by_probability(perceived_colour_value_counts)
    input4 = select_word_by_probability(perceived_colour_master_counts)
    input5 = select_word_by_probability(index_counts)
    input6 = select_word_by_probability(garment_group_counts)
    
    def find_trends():
        from bs4 import BeautifulSoup
        import requests

        trend_spotter_aw23 = requests.get('https://www.thetrendspotter.net/category/womens-fashion-tends/').text
        trend_spotter_aut_wint_fashion = requests.get('https://www.thetrendspotter.net/accessory-trends-autumn-winter-2023/').text
        trend_spotter_aut_wint_acc = requests.get('https://www.thetrendspotter.net/fashion-trends-from-autumn-winter-2023-fashion-weeks/').text

        soup1 = BeautifulSoup(trend_spotter_aw23, 'lxml')
        h2_tags = soup1.find_all('h2')
        key_words_from_websites = []
        for h2 in h2_tags:
            key_words_from_websites.append(h2.text[3:].strip())

        soup2 = BeautifulSoup(trend_spotter_aut_wint_fashion, 'lxml')
        h2_tags = soup2.find_all('h2')
        for h2 in h2_tags:
            key_words_from_websites.append(h2.text[3:].strip())

        soup3 = BeautifulSoup(trend_spotter_aut_wint_acc, 'lxml')
        h2_tags = soup3.find_all('h2')
        for h2 in h2_tags:
            key_words_from_websites.append(h2.text[3:].strip())

        key_words_from_websites_string = ",".join(key_words_from_websites)
        return key_words_from_websites_string

    
    key_words_from_websites_string = find_trends()
    user_prompt = f"\nUser requirements: {user_input}. Make sure to include the user requirements in the generated prompt. Design the outfit entirely based on user requirements. If there is no colour given by the user, use {input4}. If there is no occassion or product type mentioned by the user, use {input1} with a graphical appearance of {input2} and {input3}. If garment_type is not mentioned use {input6} of {input5} section. The latest trends are as follows {key_words_from_websites_string}. Use the trends only for females. Not for men."

    system_prompt = '''Act as world-class a fashion designer. Give me a description of a dress which adhere to the following instructions.
    1. Limit the prompt to 400 characters by using very descriptive words. Do not exceed 400 characters. It is important to adhere to 400 characters.  
    2. Make use of these fashion trends only when it pertains to user requirement. For example, don't use mini skirts for formal wear just because mini skirts is trending. Keep in mind the attire when picking the latest fashion trends. 
    3. Give preference to user requirements. And limit the generated description to only about the dress. Nothing else. Limit it to 400 characters. 
    '''
    print(user_prompt)
    
    import openai
    openai.api_key = '<open-ai-key>'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
         messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]
    )
    generated_description = response.choices[0].message["content"]
    #print(generated_description)
    return generated_description


intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
    

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = None
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    if result is None:
        result = random.choice(list_of_intents[-3]['responses'])  # Provide a default response
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents);
    if res == "Generating an outfit for you":
        res = "#"+get_desc(msg,cust_id)
    print(res)
    return res

import random
import string
def generate_random_letters(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def image_generate(user_prompt):
    global cust_id
    global imagecount
    import openai
    import requests
    openai.api_key = '<open-ai-key>'
    response = openai.Image.create(
      prompt=user_prompt,
      n=1,
      size="256x256"
    )
    image_url = response['data'][0]['url']
    print("\n")
    print(image_url)
    response = requests.get(image_url)
    product = {}
    if response.status_code == 200:
        image_filename = generate_random_letters(4)+'.png'
        currdir = os.getcwd()
        to_store = "../static/images/products/"+image_filename
        img_path = currdir+"\\static\\images\\products\\"+image_filename
        product = {
            'image_src': to_store,
            'user_id': cust_id
        }
        with open(img_path, 'wb') as image_file:
            image_file.write(response.content)
          
    products_path = currdir + "\\Templates\\products.json"
    if os.path.exists(products_path):
        with open(products_path, 'r') as json_file:
            products = json.load(json_file).get("products", [])
    products.append(product)
    
    with open(products_path, 'w') as json_file:
        json.dump({'products': products}, json_file, indent=4)
    return image_url

from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.static_folder = 'static'
@app.route("/")
def home():
    if 'username' in session:
        return render_template("index.html")
    else:
        return redirect('/login')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    s = chatbot_response(userText)
    if s[0]=='#':
        return image_generate(s[1:])
    else:
        return s;
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = json.load(open('Templates\\userbase.json'))
        global cust_id
        for item in users.get("users"): 
            print(item)
            if item["username"] == username and item["password"] == password:
                cust_id = item["customer_id"]
                print(cust_id)
                session['username'] = username 
                return redirect('/')
        else:
            return "Invalid login"
    return render_template('login.html')

@app.route("/catalog")
def catalog():
    global cust_id
    product_items = json.load(open('Templates\\products.json'))
    products = product_items.get("products")
    return render_template('catalog.html', products=products,cust_id=cust_id)

if __name__ == "__main__":
    app.run()
    
