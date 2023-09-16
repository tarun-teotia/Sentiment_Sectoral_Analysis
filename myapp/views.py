from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import csv
import requests
from bs4 import BeautifulSoup
from pathlib import Path 
from .models import collection
import datetime

# Create your views here.
def scrap(sect):
    # Create lists to store scraped news urls, headlines and text
    url_list = []
    #news_text = []
    headlines = [] 

    url = 'https://economictimes.indiatimes.com/industry/{}'.format(sect)
    n_url = 'https://economictimes.indiatimes.com'
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")
    for links in soup.find_all('div', {'class': 'top-news'}):
        for info in links.find_all('a'):
            if info.get('href') not in url_list:
                url_list.append(info.get('href'))

    for www in url_list:
        headlines.append(www.split("/")[-3].replace('-',' '))
        # request = requests.get(n_url + www)
        # soup = BeautifulSoup(request.text, "html.parser")
        # for news in soup.find_all('div', {'class': 'artText'}):
        #     news_text.append(news.text)

    # save news text along with the news headline in a dataframe
    x = {'Headline': headlines, 'sector':sect }
    news_df = pd.DataFrame(x)
        
    # export the news data into a csv file 
    filepath = Path('D:/college/7th sem project/ssa_resume_withMongo/news.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    news_df.to_csv(filepath, index=False)
    return     


def open_csv():
    # csv file name
    filename = "D:/college/7th sem project/ssa_resume_withMongo/news.csv"
    
    rows = [] 
    with open(filename, 'r', encoding = "utf-8") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row[0])
            sec = row[1]
        
    rows.pop(0)
    return [rows, sec]

#--------------------------------------------------
def i(collection, dic):
    collection.insert_one(dic)
#--------------------------------------------------

def prediction(rows):
    #############Importing trained classifier and fitted vectorizer################
    rf_clf = pickle.load(open("D:/college/7th sem project/ssa_resume_withMongo/rf_clf_6thsem", 'rb'))
    vectorizer = pickle.load(open("D:/college/7th sem project/ssa_resume_withMongo/vectorizer_6thsem", 'rb'))


    ##############Predict sentiment using the trained classifier###################
    # Import test data set

    data_pred = pd.read_csv("D:/college/7th sem project/ssa_resume_withMongo/news.csv", encoding = "ISO-8859-1")
    X_test = data_pred.iloc[:,0] # extract column with news articl
    X_vec_test = vectorizer.transform(X_test) #don't use fit_transform here because the model is already fitted
    X_vec_test = numpy.asarray(X_vec_test.todense()) #convert sparse matrix to dense

    #y_test = data_pred.iloc[:,2]

    # Transform data by applying term frequency inverse document frequency (TF-IDF) 
    tfidf = TfidfTransformer() #by default applies "l2" normalization
    X_tfidf_test = tfidf.fit_transform(X_vec_test)
    X_tfidf_test = numpy.asarray(X_tfidf_test.todense())

    # Predict the sentiment values
    y_pred = rf_clf.predict(X_tfidf_test)

    return dict(zip(rows, y_pred))


def index(request):
    if request.method == 'GET':

        if request.GET.get('1auto'):
            scrap('auto')

            return render(request, 'homepage.html', {'news_list': (open_csv())[0], 'sector': 'AUTO'})


        if request.GET.get('2energy'):
            scrap('energy')

            return render(request, 'homepage.html', {'news_list': (open_csv())[0], 'sector': 'ENERGY'})
        
        
        if request.GET.get('3transport'):
            scrap('transportation')

            return render(request, 'homepage.html', {'news_list': (open_csv())[0], 'sector': 'TRANSPORTATION'})
        
        
        if request.GET.get('4health'):
            scrap('healthcare/biotech')

            return render(request, 'homepage.html', {'news_list': (open_csv())[0], 'sector': 'HEALTHCARE'})

        
        if request.GET.get('show'):
            total = prediction((open_csv())[0])

#----------------------------------------------------------------
            l = 0
            per = 0
            obj = dict()
            
            current_time = datetime.datetime.now()
            obj['time'] = str(current_time.time())

            for k in total.values():
                if k == 'positive':
                    l += 1
            per = (l/len(total.values()))*100
            sec = (open_csv())[1]

            obj[sec] = [len(total.values()), per]
            i(collection, obj)
#----------------------------------------------------------------

            return render(request, 'homepage.html', {'all': total})


    return render(request, 'homepage.html')