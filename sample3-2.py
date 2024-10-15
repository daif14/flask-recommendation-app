import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# スケーリングする特徴量
FEATURES = ['key', 'energy', 'mode', 'acousticness', 'danceability', 'valence', 'instrumentalness', 'speechiness', 'loudness', 'tempo']

# 各ジャンルのCSVファイル名のリスト
genres = ['pop', 'rock', 'hip-hop', 'jazz', 'edm', 'reggae', 'metal']

# すべてのジャンルに対して処理を行う
for genre in genres:
    input_file = f'spotify_{genre}_features.csv'  # 入力ファイル名
    output_file = f'scaled_spotify_{genre}_features.csv'  # 出力ファイル名

    # ファイルが存在するか確認
    if not os.path.exists(input_file):
        print(f"ファイル {input_file} が見つかりません。スキップします。")
        continue

    # データの読み込み
    df = pd.read_csv(input_file)

    # データの内容を確認
    print(f"\n{genre}ジャンルの元データの最初の5行:")
    print(df.head())

    # 欠損値の確認
    print(f"\n{genre}ジャンルの欠損値の確認:")
    print(df.isnull().sum())

    # 必要な特徴量の選定
    features = df[FEATURES]

    # 特徴量の内容を確認
    print(f"\n{genre}ジャンルの選定した特徴量の最初の5行:")
    print(features.head())

    # 特徴量のスケーリング
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # スケーリング後のデータをデータフレームに保存
    df_scaled = pd.DataFrame(features_scaled, columns=FEATURES)

    # 元のトラック情報を追加（トラック名やアーティスト名、id）
    df_scaled['track_name'] = df['track_name']
    df_scaled['artist_name'] = df['artist_name']
    df_scaled['id'] = df['id']

    # スケーリング後のデータをCSVに保存
    df_scaled.to_csv(output_file, index=False)

    print(f"\n{genre}ジャンルのスケーリング後のデータの最初の5行:")
    print(df_scaled.head())

    print(f"{output_file} に保存されました。")
