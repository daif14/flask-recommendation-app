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
REDIRECT_URI = 'https://flask-recommendation-app.onrender.com'

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
FEATURES = ['mode', 'acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'speechiness']

# ジャンルごとのCSVファイルの読み込み関数
def load_genre_data(genre):
    try:
        df_genre = pd.read_csv(f'scaled_spotify_{genre}_features.csv')
        return df_genre
    except FileNotFoundError:
        return None

# Spotify認証フロー
def create_spotify_oauth():
    return SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)

@app.route('/')
def index():
    # 認証されているか確認
    if 'token_info' in session:
        print('User is already authenticated.')
        return render_template('index3.html')
    else:
        print('User is not authenticated, redirecting to login.')
        return redirect(url_for('login'))

@app.route('/login')
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    print(f'Redirecting to Spotify login: {auth_url}')
    return redirect(auth_url)

@app.route('/callback')
def callback():
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    session.clear()
    try:
        # クエリパラメータに'code'が含まれているか確認
        code = request.args.get('code')
        if code is None:
            print('No code received. Query parameters:', request.args)  # デバッグ情報
            return redirect(url_for('login'))
        
        print(f'Received code: {code}')
        token_info = sp_oauth.get_access_token(code)
        session['token_info'] = token_info
    except Exception as e:
        print(f'Error during token retrieval: {e}')
        return redirect(url_for('login'))
    
    return redirect(url_for('index'))

def get_spotify_client():
    token_info = session.get('token_info', None)
    if not token_info:
        print('No token_info found in session, redirecting to login.')
        return redirect(url_for('login'))

    # トークンの有効期限が切れていないか確認し、更新する
    sp_oauth = create_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        print('Token has expired, refreshing...')
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info  # 更新したトークンをセッションに保存
        print(f'Token refreshed: {token_info}')

    return spotipy.Spotify(auth=token_info['access_token'])


# Spotify APIを使用してユーザの再生履歴を取得
def get_user_recent_tracks():
    sp = get_spotify_client()
    recent_tracks = sp.current_user_recently_played(limit=50)  # 直近50曲を取得
    track_ids = [item['track']['id'] for item in recent_tracks['items']]
    
    # トラックの特徴量を取得
    features = sp.audio_features(tracks=track_ids)
    track_info = []
    
    for feature, item in zip(features, recent_tracks['items']):
        if feature:
            track_info.append({
                'track_name': item['track']['name'],
                'artist_name': item['track']['artists'][0]['name'],
                'id': item['track']['id'],
                'mode': feature['mode'],
                'acousticness': feature['acousticness'],
                'danceability': feature['danceability'],
                'energy': feature['energy'],
                'valence': feature['valence'],
                'instrumentalness': feature['instrumentalness'],
                'speechiness': feature['speechiness']
            })
    
    return pd.DataFrame(track_info), track_ids  # IDも返す

# ユーザの再生履歴をスケーリングする関数
def scale_user_history(user_history_df):
    # 必要な特徴量を選択
    user_features = user_history_df[FEATURES]
    
    # スケーリング
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    return pd.DataFrame(user_features_scaled, columns=FEATURES)

# コサイン類似度に基づき、推薦楽曲を取得する関数
def recommend_songs_for_user(user_scaled_features, genre_data, genre, excluded_ids):
    # ジャンルの特徴量データ
    genre_features = genre_data[FEATURES]
    
    # コサイン類似度の計算
    cosine_sim = cosine_similarity(user_scaled_features, genre_features)
    
    # 類似度の高い楽曲を上位5件取得
    sim_scores = cosine_sim[0]  # 最初のユーザ楽曲との類似度のみ使用
    top_indices = sim_scores.argsort()[::-1]  # 類似度の高い順にソート
    
    recommendations = []
    for idx in top_indices:
        track_id = genre_data.iloc[idx]['id']
        if track_id not in excluded_ids:  # 除外リストにない曲だけ
            recommendation = genre_data[['track_name', 'artist_name', 'id']].iloc[idx]
            recommendation['source_genre'] = genre
            recommendation['track_url'] = f"https://open.spotify.com/track/{track_id}"
            recommendations.append(recommendation)
            excluded_ids.add(track_id)
        
        # 推薦する楽曲は最大5件とする
        if len(recommendations) >= 5:
            break
    
    return pd.DataFrame(recommendations)

# 他のジャンルから1曲ずつ推薦を行う関数
def recommend_from_other_genres(user_scaled_features, excluded_genre, excluded_ids):
    genres = ['pop', 'rock', 'hip-hop', 'jazz', 'edm']
    genres.remove(excluded_genre)  # ユーザー指定のジャンルは除く
    recommendations = []

    for genre in genres:
        genre_data = load_genre_data(genre)
        if genre_data is not None and not genre_data.empty:
            genre_features = genre_data[FEATURES]
            cosine_sim = cosine_similarity(user_scaled_features, genre_features)
            sim_scores = cosine_sim[0]
            top_indices = sim_scores.argsort()[::-1]  # 類似度の高い順にソート
            
            for idx in top_indices:
                track_id = genre_data.iloc[idx]['id']
                if track_id not in excluded_ids:  # 除外リストにない曲だけ
                    recommendation = genre_data[['track_name', 'artist_name', 'id']].iloc[idx]
                    recommendation['source_genre'] = genre
                    recommendation['track_url'] = f"https://open.spotify.com/track/{track_id}"
                    recommendations.append(recommendation)
                    excluded_ids.add(track_id)
                    break  # 各ジャンルから1曲だけ推薦する
    
    return pd.DataFrame(recommendations)

# 楽曲推薦API
@app.route('/recommend', methods=['GET'])
def recommend():
    # ユーザが指定したジャンルを取得
    genre = request.args.get('genre')
    
    # 指定されたジャンルのデータをロード
    genre_data = load_genre_data(genre)
    if genre_data is None:
        return jsonify({"error": "ジャンルのデータが見つかりません"}), 400

    # ユーザの再生履歴を取得
    user_history_df, track_ids = get_user_recent_tracks()  # 除外リストのIDも取得
    if user_history_df.empty:
        return jsonify({"error": "再生履歴が見つかりませんでした"}), 400

    # 再生履歴の特徴量をスケーリング
    user_scaled_features = scale_user_history(user_history_df)
    
    # すでに聴いた曲のIDをセットに変換
    excluded_ids = set(track_ids)

    # ジャンルごとの楽曲から推薦を取得
    genre_recommendations = recommend_songs_for_user(user_scaled_features, genre_data, genre, excluded_ids)
    
    # 他のジャンルから1曲ずつ推薦
    other_genre_recommendations = recommend_from_other_genres(user_scaled_features, genre, excluded_ids)
    
    # 全ての推薦結果を結合
    all_recommendations = pd.concat([genre_recommendations, other_genre_recommendations])
    
    return all_recommendations.to_json(orient='records')

@app.route('/create_playlist', methods=['GET'])
def create_playlist():
    selected_track_id = request.args.get('selected_track_id')
    
    # 選択した曲の特徴量を取得
    selected_track_feature = sp.audio_features([selected_track_id])[0]
    
    # 他の楽曲の中からコサイン類似度に基づいて推薦
    all_genres = ['pop', 'rock', 'hip-hop', 'jazz', 'edm']
    playlist = []
    excluded_ids = set([selected_track_id])  # 選択された曲は除外
    
    for genre in all_genres:
        genre_data = load_genre_data(genre)
        if genre_data is not None and not genre_data.empty:
            genre_features = genre_data[FEATURES]
            
            # コサイン類似度の計算
            selected_track_features_df = pd.DataFrame([selected_track_feature], columns=FEATURES)
            cosine_sim = cosine_similarity(selected_track_features_df, genre_features)
            
            sim_scores = cosine_sim[0]
            top_indices = sim_scores.argsort()[::-1]  # 類似度の高い順にソート
            
            for idx in top_indices:
                track_id = genre_data.iloc[idx]['id']
                if track_id not in excluded_ids:  # 重複チェック
                    recommendation = genre_data[['track_name', 'artist_name', 'id']].iloc[idx]
                    recommendation['source_genre'] = genre
                    recommendation['track_url'] = f"https://open.spotify.com/track/{track_id}"
                    playlist.append(recommendation)
                    excluded_ids.add(track_id)  # 選んだ曲を除外リストに追加
                    break  # 各ジャンルから1曲だけ追加
    
    # プレイリストをDataFrameに変換してからJSONに変換
    playlist_df = pd.DataFrame(playlist)
    return playlist_df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True, port=8888)