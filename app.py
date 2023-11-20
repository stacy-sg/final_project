import csv
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
from math import pi
import plotly.express as px
import plotly
import json
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 팀별 경기 데이터를 분석하여 통계를 계산하는 함수
def calculate_team_statistics(df):
     # 팀별 경기 데이터를 분석하여 통계를 계산
    team_stats = {}

    for team in df['team_name'].unique():
        team_data = df[df['team_name'] == team]
        
        # 평균 득점, 평균 실점 계산
        avg_points_scored = team_data['total'].mean()
        avg_points_allowed = df[df['team_name'] != team]['total'].mean()  # 대략적인 계산, 실제로는 상대팀 데이터를 정확히 분석해야 함

        # 승리 횟수 계산
        wins = team_data['win_lose'].sum()
        
        # 전체 경기 수 계산
        total_games = len(team_data)

        # 승률 계산
        win_rate = wins / total_games

        team_stats[team] = {
            'avg_points_scored': avg_points_scored,
            'avg_points_allowed': avg_points_allowed,
            'win_rate': win_rate
        }

    return team_stats

def predict_matches(game_id):
    data_path = '/home/seulgi/final_project/data/kbl_game_data(+newest_team).csv'
    future_data_path = '/home/seulgi/final_project/data/kbl_game_schedule.csv'
    # 데이터셋 로드 및 처리
    df = pd.read_csv(data_path, encoding='cp949')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df.sort_values(by='game_date', inplace=True)
    
    # 팀 통계 계산 및 변환
    team_stats = calculate_team_statistics(df)
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    team_stats_df.rename(columns={'index': 'team_name'}, inplace=True)

    # 데이터 결합
    df = df.merge(team_stats_df, on='team_name', how='left')

    # 훈련 데이터 준비
    X = df[['home_away', 'avg_points_scored', 'avg_points_allowed']]
    y = df['win_lose']
    train_size = len(X) - 5
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # 모델 훈련
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 미래 경기 데이터셋 로드 및 처리
    future_games = pd.read_csv(future_data_path, encoding='utf-8')
    future_games = future_games.merge(team_stats_df, on='team_name', how='left')

    # 특정 game_id에 대한 예측
    game_predictions = []
    game_data = future_games[future_games['game_id'] == game_id]
    if not game_data.empty:
        team_1_name = game_data.iloc[0]['team_name']
        team_2_name = game_data.iloc[1]['team_name']

        # 각 팀의 최근 5경기 데이터 추출
        recent_data_team_1 = df[(df['team_name'] == team_1_name) & (df['game_date'] < game_data.iloc[0]['game_date'])].tail(5)
        recent_data_team_2 = df[(df['team_name'] == team_2_name) & (df['game_date'] < game_data.iloc[0]['game_date'])].tail(5)

        # 필요한 피처 선택 및 데이터프레임 생성
        features_team_1 = recent_data_team_1[['home_away', 'avg_points_scored', 'avg_points_allowed']].mean().rename(lambda x: f'team_1_{x}')
        features_team_2 = recent_data_team_2[['home_away', 'avg_points_scored', 'avg_points_allowed']].mean().rename(lambda x: f'team_2_{x}')
        X_match = pd.concat([features_team_1, features_team_2], axis=1)

        # 예측 수행
        prediction = model.predict_proba(X_match)
        team_1_win_probability = prediction[0][1]
        team_2_win_probability = prediction[0][0]

        # 결과 추가
        game_predictions.append({
            'game_id': game_id, 
            'team_1': team_1_name,
            'team_1_win_probability': team_1_win_probability, 
            'team_2': team_2_name,
            'team_2_win_probability': team_2_win_probability,
        })

    return game_predictions

def radar_chart(season, player_name):
    # CSV 파일 경로 설정
    file_path = f"/home/seulgi/final_project/data/kbl_player_data_{season}.csv"

    # CSV 파일 불러오기
    df = pd.read_csv(file_path, encoding='cp949')

    # 레이더 차트에 사용할 지표 선택
    categories = ['FG', 'FT', 'STL', 'TOT-REBOUND', 'AST', 'BS']

    # MinMaxScaler 객체 생성 및 정규화 수행
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[categories] = scaler.fit_transform(df[categories])

    # 특정 선수 데이터 추출
    player_data_normalized = df_normalized[df_normalized['name'] == player_name]

    if player_data_normalized.empty:
        print("해당 선수의 데이터가 없습니다.")
        return

    # 각 지표별 값 추출
    values = player_data_normalized.iloc[0][categories].tolist()
    values += values[:1]  # 원 완성을 위해 첫 값을 마지막에 추가

    # 각 축의 각도 계산
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    # 레이더 차트 설정
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.plot(angles, values)
    ax.fill(angles, values, 'teal', alpha=0.1)

    # 차트 타이틀 설정
    plt.title(f'{season} Season - Performance Radar Chart')

    return fig

def player_stats(season, player_name):
    # CSV 파일 경로 설정
    file_path = f"/home/seulgi/final_project/data/kbl_player_data_{season}.csv"

    # CSV 파일 불러오기
    df = pd.read_csv(file_path, encoding='cp949')

    # 성공률 관련 컬럼 제외
    exclude_columns = ['FG%', 'FT%', 'PP%']
    data_columns = [col for col in df.columns if col not in exclude_columns and col not in ['season', 'name', 'team']]

    # 특정 선수 데이터 추출
    player_data = df[df['name'] == player_name]

    # 데이터가 없는 경우 처리
    if player_data.empty:
        print(f"{season} 시즌의 {player_name} 선수의 데이터가 없습니다.")
        return

    # 각 컬럼에 대해 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, len(data_columns) * 2))  # 그래프를 위한 figure 생성
    spacing = 2  # 간격 조절

    for i, col in enumerate(data_columns):
        y = i * spacing
        ax.scatter(df[col], [y]*len(df), color='gray', label='All Players' if i == 0 else "", s=50)
        if not player_data[col].isna().values.any():
            player_value = player_data[col].values[0]
            ax.scatter(player_value, y, color='skyblue', label='Selected Player' if i == 0 else "", s=100)
            ax.text(player_value, y + 0.1, f'{player_value:.2f}', color='black', va='bottom', ha='center')

    ax.set_yticks([i * spacing for i in range(len(data_columns))])
    ax.set_yticklabels(data_columns)
    ax.legend(loc='upper right')
    return fig

def field_goal_performance(season, player_name):
    # CSV 파일 경로 설정
    file_path = f"/home/seulgi/final_project/data/kbl_player_data_{season}.csv"

    # CSV 파일 불러오기
    df = pd.read_csv(file_path, encoding='cp949')

    # 스캐터 플롯 생성
    fig = px.scatter(df, x="FG", y="FGA", hover_data=['name', 'team', 'season'],
                     title='Field Goal Attempts vs Success',
                     labels={'FGA': 'Field Goal Attempts', 'FG': 'Field Goal Success'})

    # 특정 선수의 데이터 강조
    player_data = df[df['name'] == player_name]
    if not player_data.empty:
        fig.add_scatter(x=player_data['FG'], y=player_data['FGA'], mode='markers',
                        marker=dict(color='skyblue', size=10), name=player_name)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def rebounds(season, player_name):
    # CSV 파일 경로 설정
    file_path = f"/home/seulgi/final_project/data/kbl_player_data_{season}.csv"

    # CSV 파일 불러오기
    df = pd.read_csv(file_path, encoding='cp949')

    # 스캐터 플롯 생성
    fig = px.scatter(df, x="OFF- REBOUND", y="TOT-REBOUND", hover_data=['name', 'team', 'season'],
                     title='Offense-Rebound vs Total-Rebound',
                     labels={'OFF- REBOUND': 'Offense-Rebound', 'TOT-REBOUND': 'Total-Rebound'})

    # 특정 선수의 데이터 강조
    player_data = df[df['name'] == player_name]
    if not player_data.empty:
        fig.add_scatter(x=player_data['OFF- REBOUND'], y=player_data['TOT-REBOUND'], mode='markers',
                        marker=dict(color='skyblue', size=10), name=player_name)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

app = Flask(__name__)

@app.route('/')
def home():
    # 홈 화면을 렌더링합니다.
    return render_template('home.html')

@app.route('/match_prediction')
def match_prediction():
    # 경기 결과 예측 화면을 렌더링합니다.
    # 필요한 데이터를 데이터베이스에서 가져와서 전달할 수 있습니다.
    return render_template('match_prediction.html')

@app.route('/player_statistics')
def player_statistics():
    # 선수 데이터 통계 화면을 렌더링합니다.
    # 필요한 데이터를 데이터베이스에서 가져와서 전달할 수 있습니다.
    return render_template('player_statistics.html')

@app.route('/get-game-schedule', methods=['POST'])
def get_game_schedule():
    try:
        year = int(request.form['year'])
        month = int(request.form['month'])

        csv_file = "/home/seulgi/final_project/data/kbl_game_schedule.csv"
        # CSV 파일 로드 및 날짜 형식 변환
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        df['game_date'] = pd.to_datetime(df['game_date'])

        # 년도와 월에 맞는 데이터 필터링
        mask = (df['game_date'].dt.year == year) & (df['game_date'].dt.month == month)
        filtered_games = df[mask]

        # 필요한 데이터만 추출
        games_data = filtered_games[['game_id', 'game_date', 'team_name', 'home_away']]  # 필요한 컬럼 지정
        # games_data를 리스트로 변환
        games_data_list = games_data.to_dict(orient='records')

        # JSON 형식으로 응답
        return jsonify(games_data_list)
    except Exception as e:
        return jsonify({'error': str(e)})

# 경기 결과를 예측하는 라우트
@app.route('/predict-game', methods=['GET'])
def predict_game_route():
    game_id = request.args.get('game_id')
    if game_id:
        try:
            game_id = int(game_id)  # game_id를 정수로 변환
            predictions = predict_matches(game_id)
            predictions = predictions.to_dict(orient='records')
            return jsonify(predictions)
        except ValueError:
            return jsonify({"error": "Invalid game_id"}), 400
    else:
        return jsonify({"error": "game_id parameter is required"}), 400

@app.route('/get-player-data', methods=['GET'])
def get_player_data():
    season = request.args.get('season')
    player_name = request.args.get('player')

    # CSV 파일 경로 구성 (season 값에 기반하여)
    csv_file_path = f'/home/seulgi/final_project/data/kbl_player_data_{season}.csv'

    # CSV 파일을 열고 데이터를 찾음
    try:
        with open(csv_file_path, newline='', encoding='cp949') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 선수 이름이 일치하는 데이터를 찾음
                if row['name'] == player_name:
                    # 찾은 데이터를 JSON 형식으로 반환
                    return jsonify(row)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    
    # 일치하는 선수가 없는 경우
    return jsonify({'error': 'Player not found'}), 404

# 데이터를 제공하는 API 라우트
@app.route('/api/all-players')
def all_players():
    season = request.args.get('season')
    csv_file_path = f'/home/seulgi/final_project/data/kbl_player_data_{season}.csv'

    try:
        with open(csv_file_path, newline='', encoding='cp949') as file:
            csv_reader = csv.DictReader(file)
            players = [row for row in csv_reader]
        return jsonify(players)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/plot/radar-chart/<season>/<player_name>')
def radar_chart_image(season, player_name):
    fig = radar_chart(season, player_name)
    if fig:
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        fig.clf()  # 클리어 figure
        plt.close(fig)  # figure 닫기
        return send_file(img, mimetype='image/png')
    else:
        return jsonify({"error": "Image not generated"}), 500

@app.route('/plot/player-stats/<season>/<player_name>')
def player_stats_image(season, player_name):
    fig = player_stats(season, player_name)  # 올바른 함수를 호출
    if fig:
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')  # 수정: fig 객체에 대한 savefig 호출
        img.seek(0)
        fig.clf()  # 클리어 figure
        plt.close(fig)  # figure 닫기
        return send_file(img, mimetype='image/png')
    else:
        return jsonify({"error": "Image not generated"}), 500

@app.route('/plot/field-goal-performance/<season>/<player_name>')
def field_goal_performance_image(season, player_name):
    graphJSON = field_goal_performance(season, player_name)
    return graphJSON

@app.route('/plot/rebounds/<season>/<player_name>')
def rebounds_image(season, player_name):
    graphJSON = rebounds(season, player_name)
    return graphJSON

if __name__ == '__main__':
    app.run(debug=True)
