#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy import displacy
import pandas as pd
import re
import itertools as it
import os
import random
from spacy.training import offsets_to_biluo_tags,biluo_tags_to_spans
from spacy.tokens import Doc, DocBin
from ast import literal_eval


# In[2]:


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


# In[59]:


file=pd.read_csv("RAW_recipes.csv")
file.head(5)


# In[60]:


#preprocessing ingredients and raw recipies columns
#converting ingredients to a list of lists, then flattening the ingredients list, to create a single list containing all ingredients
file["ingredients"]= file["ingredients"].apply(lambda x: literal_eval(x))
ingredients= file["ingredients"].tolist()
ingredients


# In[61]:


ingredients = list(set(it.chain(*ingredients)))#flattening the list, then converting it to a set to remove duplicates, and converting it to a list again
ingredients


# In[62]:


#doing the same with steps
file["steps"]= file["steps"].apply(lambda x: literal_eval(x))
steps = file["steps"].tolist()
steps = list(set(it.chain(*steps)))
steps


# In[63]:


#next we define a pattern, and train the nlp model with our defined pattern
pattern=re.compile(r'\b(?:%s)\b'% '|'.join(ingredients))
pattern


# In[64]:


#next we create annotations, and search for pattern inside the text
annotations = []
annotation_text = []
for i in steps:
    match = pattern.finditer(i) #finditer gives all words match along with the span
    #next we create dictionaries
    temp1 = {}
    temp2 = {}
    val1 = []
    val2 = []
    for m in match:
        if m.group():
            val1.append([m.start(), m.end(), "Ingredients"]) #to get the start and end of each word span and append it to one list
            val2.append([m.group(),m.start(), m.end(), "Ingredients"])  # eher we append the llist with word and span, this is not required
        temp1.update({"Entities":val1}) #next we update our dictionary with the created files
        temp2.update({"Entities":val2})
        annotations.append([i, temp1])
        annotation_text.append([i,temp2])


# In[65]:


annotation_text


# In[68]:


db = DocBin() #for serializing the model, to speed up the training process, by packing everything  into one object
#we split the dataset to train and val, tehn we seperate the text lines, and annotations
for text, annot in annotations[0:10000]: #keeping 1000 for testing
    try:
        docs = nlp.make_doc(text) #converting the text strings to normal one documnet
        tags = offsets_to_biluo_tags(docs, annot["Entities"]) # we add the bilou tags to words
        ents = biluo_tags_to_spans(docs, tags) # then find their spans
        docs.ents = ents
        db.add(docs)
    except IndexError:
        pass #if any statement is giving error pass will not consider that
    db.to_disk("train.spacy") 


# In[70]:


for text, annot in annotations[10000:12000]: #creating for val
    try:
        doc = nlp.make_doc(text)
        tags = offsets_to_biluo_tags(doc, annot["Entities"]) # we add the bilou tags to words
        ents = biluo_tags_to_spans(doc, tags) # then find their spans
        doc.ents = ents
        db.add(doc)
    except IndexError:
        pass #if any statement is giving error pass will not consider that
    db.to_disk("validation.spacy")


# In[3]:


#now we use transsfer learning
get_ipython().system(' python -m spacy init fill-config base_config.cfg config.cfg')


# In[3]:


get_ipython().system(' python -m spacy debug data config.cfg')


# In[6]:


get_ipython().system(' python -m spacy train "C:/Users/lexus/Desktop/DGI/week5/config.cfg" --output ./output')


# In[9]:


#loading the model
recepie_model = spacy.load("C:/Users/lexus/Desktop/DGI/week5/output/model-best")


# In[13]:


#predicting the ingredients
recepie1 = "in a medium frying pan , cook the bacon over medium heat until cooked but not crispy , 10 to 12 minutes , stirring frequently', 'remove the bacon to a paper towel-lined plate or pan , to remove excess fat', 'in a large bowl , whisk together the flour , sugar , baking powder , baking soda and salt', 'using a pastry cutter or fork , cut in the diced butter , until it resembles small peas', 'stir in the bacon , then one-fourth cup plus 2 tablespoons of maple syrup and the buttermilk until the dough just comes together', 'be careful not to overwork the dough', 'on a lightly floured surface , gently press or roll the dough to 1-inch thickness', 'cut the biscuits using a 2-inch round cutter', 'you should have 24 biscuits', 'place 12 biscuits on each of two parchment-lined baking sheets , spaced 2 inches apart', 'freeze the trays just until the biscuits are chilled , about 10 minutes', 'heat the oven to 350 degrees', 'while the biscuits are chilling , prepare the egg wash: in a small bowl , whisk together the egg yolk , egg and cream', 'brush the chilled biscuits with egg wash and top each with a pinch of fleur de sel', 'bake the biscuits until they just begin to brown , about 25 minutes', 'remove the tray from the oven', 'quickly drizzle 1 teaspoon of the remaining maple syrup over each biscuit , then place the tray back in the oven for 3 minutes more', 'bake from frozen: heat oven to 400 degrees', 'place desired number of frozen biscuits on a lightly oiled baking sheet', 'bake for 20-26 minutes to desired browning is achieved"
doc1 = recepie_model(recepie1)
ingredients1 = [ent.text for ent in doc1.ents]
print(recepie1, "\n\n", "Ingredients:", ingredients1)


# In[15]:


#loading the pickle model
import pickle
#saving the model as a pickle file
pickle.dump(recepie_model, open('model.pkl','wb'))


# In[16]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
recepie2 = "saute black walnuts in the 3 tbs', 'of butter & let cool', 'combine sugar , flour & salt , slowly stir in hot milk', 'cook 10 minutes over low heat , stirring constantly', 'stir small amount of cooked mixture into beaten eggs , return to milk mixture & cook 1 mniute', 'chill in refrigerator , then pour in ice cream freezer & add cream & vanilla', 'churn in freezer for about 15 minutes , then add black walnuts & finish freezing"
doc2 = model(recepie2)
ingredients2 = [ent.text for ent in doc2.ents]
print(recepie2, "\n\n", "Ingredients:", ingredients2)


# In[ ]:




