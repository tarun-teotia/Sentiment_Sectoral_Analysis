from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import csv
import requests
from bs4 import BeautifulSoup
from pathlib import Path 

# Create your views here.
def scrap(sect):
    # Create lists to store scraped news urls, headlines and text
    url_list = []
    news_text = []
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
        request = requests.get(n_url + www)
        soup = BeautifulSoup(request.text, "html.parser")
        for news in soup.find_all('div', {'class': 'artText'}):
            news_text.append(news.text)

    # save news text along with the news headline in a dataframe
    if sect == 'transportation':
        headlines.pop(8)

    x = { 'Headline': headlines, 'News': news_text, }
    news_df = pd.DataFrame(x)
        
    # export the news data into a csv file 
    filepath = Path('D:/college/3rd year/6thsemproject/news.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    news_df.to_csv(filepath, index=False)
    return     


def open_csv():
    # csv file name
    filename = "D:/college/3rd year/6thsemproject/news.csv"
    
    rows = [] 
    with open(filename, 'r', encoding = "utf-8") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row[0])
        
    rows.pop(0)
    return rows


def prediction(rows):
    #############Importing trained classifier and fitted vectorizer################
    rf_clf = pickle.load(open("D:/college/3rd year/6thsemproject/rf_clf_6thsem", 'rb'))
    vectorizer = pickle.load(open("D:/college/3rd year/6thsemproject/vectorizer_6thsem", 'rb'))


    ##############Predict sentiment using the trained classifier###################
    # Import test data set

    data_pred = pd.read_csv("D:/college/3rd year/6thsemproject/news.csv", encoding = "ISO-8859-1")
    X_test = data_pred.iloc[:,1] # extract column with news articl
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
        rows = []


        if request.GET.get('1auto'):
            scrap('auto')

            return render(request, 'homepage.html', {'news_list': open_csv(), 'sector': 'AUTO'})


        if request.GET.get('2energy'):
            scrap('energy')

            return render(request, 'homepage.html', {'news_list': open_csv(), 'sector': 'ENERGY'})
        
        
        if request.GET.get('3transport'):
            scrap('transportation')

            return render(request, 'homepage.html', {'news_list': open_csv(), 'sector': 'TRANSPORTATION'})
        
        
        if request.GET.get('4health'):
            scrap('healthcare/biotech')

            return render(request, 'homepage.html', {'news_list': open_csv(), 'sector': 'HEALTHCARE'})

        
        if request.GET.get('show'):
            total = prediction(open_csv())

            return render(request, 'homepage.html', {'all': total})


    return render(request, 'homepage.html')