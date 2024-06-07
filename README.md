Shaun's Spotify Recommender System
Project Overview
Welcome to my Spotify Recommender System! This project represents a blend of my love for music and my passion for data analysis. Growing up in a household where music was the heartbeat of our daily lives, I’ve always been curious about exploring new tracks and artists. This project aims to answer a simple yet intriguing question: "Who and what does Shaun listen to the most on Spotify?"

Table of Contents
Introduction
Project Goals
Data Collection
Data Analysis
Recommendations
Cluster Analysis
Audio Features Analysis
Playlist Creation
Future Work
Usage Instructions
Contact

Introduction
This project leverages the Spotify API to fetch my liked songs, analyze them, and provide personalized recommendations. It showcases my ability to connect to APIs, clean and analyze data, and visualize insights.

Project Goals
Understand my musical preferences: Identify the genres, artists, and characteristics that dominate my playlist.
Build a recommendation system: Suggest new tracks based on my listening history.
Analyze audio features: Gain insights into the audio properties that appeal to me the most.
Create a dynamic playlist: Automatically update my recommendations with a weekly cron job.

Data Collection
Spotify API
Using the Spotify API, I extracted data for over 3,700 liked songs, including:

Track names
Artists
Albums
Popularity scores
Audio features (e.g., danceability, energy, tempo, etc.)
python

Copy code
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    redirect_uri='YOUR_REDIRECT_URI',
    scope='user-library-read user-top-read playlist-modify-public'
))

# Fetch liked songs
results = sp.current_user_saved_tracks(limit=50)
liked_songs = results['items']

# Continue fetching more liked songs
while results['next']:
    results = sp.next(results)
    liked_songs.extend(results['items'])

Data Analysis
Genre and Feature Analysis
Analyzed the genres and audio features to understand my music preferences. Key insights include:

Dominant genres: Hip hop and rap were the most prevalent.

Audio characteristics: Tracks with high danceability and energy were preferred, indicating a penchant for lively and engaging music.

Python
Copy code
# Extract relevant features and metadata
song_data = []
for item in liked_songs:
    track = item['track']
    track_info = {
        'track_id': track['id'],
        'track_name': track['name'],
        'artist_name': track['artists'][0]['name'],
        'popularity': track['popularity'],
        'album': track['album']['name']
    }
    song_data.append(track_info)

# Convert to DataFrame
df_songs = pd.DataFrame(song_data)
Recommendations
Developed a recommendation engine using TF-IDF and cosine similarity to suggest new songs based on my current playlist. Learning about these techniques and implementing them was a rewarding challenge.

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine metadata for each song
df_songs['metadata'] = df_songs[['artist_name', 'genres', 'popularity']].apply(lambda x: ' '.join(map(str, x)), axis=1)

# Calculate TF-IDF and cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_songs['metadata'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get recommendations
def get_recommendations(track_name, similarity_df, df_songs, num_recommendations=5):
    similar_tracks = similarity_df[track_name].sort_values(ascending=False)[1:num_recommendations+1]
    similar_track_names = similar_tracks.index
    return df_songs[df_songs['track_name'].isin(similar_track_names)]
Cluster Analysis
Using PCA and KMeans clustering, I grouped my songs into distinct clusters. Each cluster represented unique musical themes, helping me understand the diversity in my playlist.

python
Copy code
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Perform PCA and KMeans clustering
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_songs[['danceability', 'energy', 'tempo', 'valence', 'acousticness']])
df_songs['pca1'] = principal_components[:, 0]
df_songs['pca2'] = principal_components[:, 1]

kmeans = KMeans(n_clusters=5, random_state=42)
df_songs['cluster'] = kmeans.fit_predict(df_songs[['danceability', 'energy', 'tempo', 'valence', 'acousticness']])
Audio Features Analysis
Conducted a detailed analysis of audio features across my liked tracks and recommended tracks. This analysis provided insights into the characteristics that define my musical taste.

python
Copy code
# Fetch audio features for tracks
def fetch_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        audio_features.extend(sp.audio_features(batch))
    return audio_features

track_ids = df_songs['track_id'].tolist()
audio_features = fetch_audio_features(sp, track_ids)

# Convert audio features to DataFrame
df_audio_features = pd.DataFrame(audio_features)
Playlist Creation
Created a new playlist with the top recommendations and set up a cron job to update it weekly based on my latest liked tracks. This dynamic playlist ensures I always have fresh music to explore.

python
Copy code
# Create a new playlist
user_id = sp.current_user()['id']
playlist_name = 'Shaun New Recommended Playlist'
new_playlist = sp.user_playlist_create(user_id, playlist_name, public=True)

# Add recommended tracks to the playlist
track_uris = refined_recommendations['track_id'].apply(lambda x: f'spotify:track:{x}').tolist()
sp.playlist_add_items(new_playlist['id'], track_uris)
Future Work
Weekly Updates: Implement a cron job to update my playlist with new recommendations based on my recent likes.
Improved Recommendations: Continuously refine the recommendation algorithm to incorporate more audio features and improve accuracy.
Usage Instructions
Clone the repository:
bash
Copy code
git clone https://github.com/shaunmckellarjr/Spotify-Recommender-System.git
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
Run the script:
bash
Copy code
python spotify_recommender.py
Contact
Feel free to reach out if you have any questions or suggestions!

Email: shaun.mckellar@example.com
LinkedIn: Shaun McKellar Jr
