import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv('data.csv')
search = pd.read_csv('query.csv')
pivot = data.pivot_table(index = 'title', columns = 'userId',values = 'rating').fillna(0)

def help(m): 
    m = m.lower()
     # check if the movie is in our database or not
    if m not in search['title'].unique():           
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
    # getting the index of the movie in the dataframe
        query_index = search.loc[search['title']==m].index[0]
        distances, indices = model.kneighbors(pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 11)
        l = []
        for i in range(1, len(distances.flatten())):
            l.append(pivot.index[indices.flatten()[i]])
        return l
        
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = help(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')

if __name__ == '__main__':
    app.run(debug=True)
