import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spotify APIの認証情報
CLIENT_ID = 'f48dda32a0544428a6808ffc4a03e5ec'
CLIENT_SECRET = '898a3fa1764d4471aa965cc8044ce02b'

# Spotify APIへの認証
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# リクエスト制限に対応するための待機時間（秒）
REQUEST_DELAY = 1
MAX_RETRIES = 3

# 各ジャンルごとのプレイリストを検索し、トラックIDを取得
def get_track_ids_by_genre(genre, tracks_per_playlist=100):
    track_ids = set()
    results = sp.search(q=f'genre:{genre}', type='playlist', limit=10)  # ジャンルごとにプレイリストを検索
    for playlist in results['playlists']['items']:
        playlist_tracks = sp.playlist_tracks(playlist['id'], limit=tracks_per_playlist)
        for item in playlist_tracks['items']:
            track_ids.add(item['track']['id'])
        logger.info(f'Genre: {genre}, Playlist ID: {playlist["id"]}, Tracks fetched: {len(track_ids)}')
        time.sleep(REQUEST_DELAY)
    return list(track_ids)

# トラックの特徴量を取得してデータフレームに保存
def get_track_features(track_ids):
    features_list = []
    track_info_list = []
    total_tracks = len(track_ids)
    for i in range(0, total_tracks, 50):
        batch = track_ids[i:i + 50]
        for attempt in range(MAX_RETRIES):
            try:
                features = sp.audio_features(batch)
                for feature in features:
                    if feature:
                        features_list.append(feature)
                        track_info = sp.track(feature['id'])
                        track_name = track_info['name']
                        artist_name = track_info['artists'][0]['name']
                        track_info_list.append({'id': feature['id'], 'track_name': track_name, 'artist_name': artist_name})
                logger.info(f'Processed {i + len(batch)} / {total_tracks} tracks')
                time.sleep(REQUEST_DELAY)
                break
            except Exception as e:
                logger.error(f'Error processing batch {i // 50 + 1}: {e}')
                time.sleep(REQUEST_DELAY * (attempt + 1))
                if attempt == MAX_RETRIES - 1:
                    logger.error(f'Failed to process batch {i // 50 + 1} after {MAX_RETRIES} attempts')
    return pd.DataFrame(features_list), pd.DataFrame(track_info_list)

# 各ジャンルに対して特徴量を取得し、CSVファイルに保存
# genres = ['pop', 'rock', 'hip-hop', 'jazz', 'edm', 'reggae', 'metal']
genres = ['j-rock']

for genre in genres:
    logger.info(f'Fetching track IDs for genre: {genre}')
    track_ids = get_track_ids_by_genre(genre)
    
    logger.info(f'Fetching track features for genre: {genre}')
    df_features, df_track_info = get_track_features(track_ids)

    # 特徴量データフレームにトラック情報を結合
    df = pd.merge(df_features, df_track_info, on='id')

    # 必要な列のみを選択
    df = df[['id', 'track_name', 'artist_name', 'acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key', 'duration_ms', 'time_signature']]

    # 各ジャンルごとにCSVファイルを保存
    csv_filename = f'spotify_{genre}_features.csv'
    df.to_csv(csv_filename, index=False)

    logger.info(f'{genre} genre features saved to {csv_filename}')
