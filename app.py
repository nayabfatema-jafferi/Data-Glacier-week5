#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request,render_template
import pickle


# In[5]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


# In[6]:


@app.route('/')
def home():
    return render_template('index.html')


# In[7]:


@app.route('/predict',methods = ["POST"])
def predict():
    recepie = request.form["recepie_steps"]
    doc = model(recepie)
    ingredients = [ent.text for ent in doc.ents]
    return render_template('index.html', prediction_text="Ingredeints: {}".format(ingredients))
    
        


# In[8]:


if __name__ == "__main__":
    app.run(debug=True)

