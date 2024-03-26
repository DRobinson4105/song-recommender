import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import TransformerMixin, BaseEstimator
from itertools import islice
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

# Spotipy setup
client_credentials_manager = SpotifyClientCredentials(
    os.environ.get('SPOTIFY_CLIENT_ID'), os.environ.get('SPOTIFY_CLIENT_SECRET'))
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# PyTorch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_features)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class MultiHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.mlb = None
        
    def fit(self, X):
        genres = [x.split() for x in X.iloc[:, 0]]
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(genres)
        return self
    
    def transform(self, X):
        genres = [x.split() for x in X.iloc[:, 0]]
        return self.mlb.transform(genres)
    
def load_model() -> Autoencoder:
    model = Autoencoder(processed_data().shape[1])
    model.load_state_dict(torch.load(
        f=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth'),
        map_location=device
    ))
    return model

def get_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset.csv'))

    # Remove duplicate tracks
    df = df.drop_duplicates(subset=['track_name', 'artist', 'genre'], keep='first')

    # Aggregate all tracks that have the same name and artist
    special_aggregations = {
        'genre': lambda x: ' '.join(x),
        'track_id': 'first'
    }
    aggregations = {col: 'mean' if col not in special_aggregations else special_aggregations[col]
                    for col in df.columns if col not in ['track_name', 'artist']}
    df = df.groupby(['track_name', 'artist']).agg(aggregations).reset_index()

    return df

def processed_data(df=get_data()):
    features = df.iloc[:, 3:]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['danceability', 'energy', 'key', 'loudness', 'mode',
                                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                                   'valence', 'tempo']),
        ('multi_hot', MultiHotEncoder(), ['genre'])
    ])

    return preprocessor.fit_transform(features)

def get_similar_songs(playlist, df=get_data(), n=5):
    playlist = [name for name in playlist if name in df["track_name"].unique()]

    if not playlist: return []

    song_names = df['track_name'].values
    song_artists = df['artist'].values
    genres = df['genre'].values
    index_to_genre = {index: genre for index, genre in enumerate(genres)}
    model = load_model()
    
    model.eval()
    with torch.inference_mode():
        tensor = torch.FloatTensor(processed_data(df))
        latent_features = model.encoder(tensor).numpy()
    
    # get common genre
    genre_count = {}
    for song_name in playlist:
        song_index = np.where(song_names == song_name)[0][0]
        if song_index is not None:
            song_genre = index_to_genre[song_index]
            genres = index_to_genre[song_index].split()
            for genre in genres:
                genre_count[genre] = genre_count.get(song_genre, 0) + 1

    common_genre = max(genre_count, key=genre_count.get)
    
    # get average features
    latent_features_list = []
    for song_name in playlist:
        song_index = np.where(song_names == song_name)[0][0]
        if song_index is not None:
            latent_features_list.append(latent_features[song_index])
            
    average_features = np.mean(latent_features_list, axis=0) if latent_features_list else None
    
    # find similar songs
    similarities = cosine_similarity([average_features], latent_features)[0]
    indices = np.argsort(-similarities)

    return [x for x in islice(((song_names[index], song_artists[index])
                            for index in indices if common_genre in df[df['track_name'] ==
                            song_names[index]]['genre'].iloc[0]), n)]

def get_spotify_songs(playlist_id):
    return [track["track"]["name"] for track in sp.playlist_tracks(playlist_id)["items"]]