머신러닝을 활용한 스포츠 경기 결과 예측과 선수 데이터 통계 시각화 서비스

데이터 크롤링을 통한 직접적인 데이터 수집, 수집한 데이터를 정형화 한 후에 XGBoost를 활용하여 경기 결과를 예측, 선수 데이터를 시즌 별로 시각화하여 시즌별 선수 검색 서비스를 제공

app.py - 최종 실행 파일로 각 실행을 위한 route 코드가 존재
home.html - 웹을 실행하면 나오는 첫 번째 화면으로 화면 구성 코드가 존재
match_prediction.html - 경기 결과 예측 화면으로 시즌과 월을 선택하면 해당하는 경기 스케줄과 예측 결과 출력 코드가 존재
player_statistics.html - 선수 데이터 통계를 시각화한 화면으로 시즌을 선택하고 원하는 선수를 검색하면 해당 선수의 기록을 시각화하여 제공하는 코드가 존재
style.css - 모든 화면의 화면 구성과 디자인을 제공하는 코드가 존재