from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

# Spotify API 認証情報
CLIENT_ID = 'f48dda32a0544428a6808ffc4a03e5ec'
CLIENT_SECRET = '898a3fa1764d4471aa965cc8044ce02b'
REDIRECT_URI = 'https://flask-recommendation-app.onrender.com/callback'

# スコープに再生履歴を含める
SCOPE = 'user-read-recently-played'

# Spotify API 認証をセットアップ
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE))

app = Flask(__name__)
app.secret_key = os.urandom(24)  # セッション用のシークレットキーを設定

# 使用する特徴量
FEATURES = ['key', 'energy', 'mode', 'acousticness', 'danceability', 'valence', 'instrumentalness', 'speechiness', 'loudness', 'tempo']

# Spotify認証フロー
def create_spotify_oauth():
    return SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)

# 複数ジャンルの楽曲データを読み込む
def load_all_genre_data():
    all_genre_data = pd.DataFrame()
    for genre in ['pop', 'rock', 'hip-hop', 'jazz', 'edm']:
        genre_data = load_genre_data(genre)
        if genre_data is not None:
            all_genre_data = pd.concat([all_genre_data, genre_data])
    return all_genre_data

def load_genre_data(genre):
    try:
        return pd.read_csv(f'scaled_spotify_{genre}_features.csv')
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    if 'token_info' in session:
        return render_template('index3.html')
    else:
        return redirect(url_for('login'))

@app.route('/login')
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/callback')
def callback():
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    session.clear()
    code = request.args.get('code')
    if code:
        token_info = sp_oauth.get_access_token(code)
        session['token_info'] = token_info
    return redirect(url_for('index'))

def get_spotify_client():
    token_info = session.get('token_info', None)
    if not token_info:
        return redirect(url_for('login'))
    sp_oauth = create_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return spotipy.Spotify(auth=token_info['access_token'])

# ユーザの再生履歴を取得
def get_user_recent_tracks():
    sp = get_spotify_client()
    recent_tracks = sp.current_user_recently_played(limit=50)
    track_ids = [item['track']['id'] for item in recent_tracks['items']]
    features = sp.audio_features(tracks=track_ids)
    track_info = [{
        'track_name': item['track']['name'],
        'artist_name': item['track']['artists'][0]['name'],
        'id': item['track']['id'],
        'key': feature['key'],
        'energy': feature['energy'],
        'mode': feature['mode'],
        'acousticness': feature['acousticness'],
        'danceability': feature['danceability'],
        'valence': feature['valence'],
        'instrumentalness': feature['instrumentalness'],
        'speechiness': feature['speechiness'],
        'loudness': feature['loudness'],
        'tempo': feature['tempo']
    } for feature, item in zip(features, recent_tracks['items']) if feature]
    
    return pd.DataFrame(track_info), track_ids

# ユーザ再生履歴の特徴量をスケーリング
def scale_user_history(user_history_df):
    user_features = user_history_df[FEATURES]
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(user_features), columns=FEATURES)

# 全ジャンルからコサイン類似度で楽曲を推薦
def recommend_top_songs(user_scaled_features, all_genre_data, excluded_ids):
    genre_features = all_genre_data[FEATURES]
    cosine_sim = cosine_similarity(user_scaled_features.mean(axis=0).values.reshape(1, -1), genre_features)
    sim_scores = cosine_sim[0]
    top_indices = sim_scores.argsort()[::-1]
    
    recommendations = []
    for idx in top_indices:
        track_id = all_genre_data.iloc[idx]['id']
        if track_id not in excluded_ids:
            recommendation = all_genre_data[['track_name', 'artist_name', 'id']].iloc[idx]
            recommendation['track_url'] = f"https://open.spotify.com/track/{track_id}"
            recommendations.append(recommendation)
            excluded_ids.add(track_id)
        if len(recommendations) >= 7:
            break
    return pd.DataFrame(recommendations)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_history_df, track_ids = get_user_recent_tracks()
    if user_history_df.empty:
        return jsonify({"error": "再生履歴が見つかりませんでした"}), 400

    user_scaled_features = scale_user_history(user_history_df)
    excluded_ids = set(track_ids)
    all_genre_data = load_all_genre_data()
    recommendations = recommend_top_songs(user_scaled_features, all_genre_data, excluded_ids)
    
    return recommendations.to_json(orient='records')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8888))
    app.run(debug=True, host='0.0.0.0', port=port)