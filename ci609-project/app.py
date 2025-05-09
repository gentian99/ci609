import time
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, g
)
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from datetime import datetime
from prediction import predict_match
import pandas as pd
from preprocessing import load_and_filter_data

#app setup
app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

startup_ts = time.perf_counter()
first_req_logged = False

#request timing & caching
@app.before_request
def start_timer():
    global first_req_logged
    if not first_req_logged:
        app.logger.info(f"Startup time: {time.perf_counter() - startup_ts:.3f}s")
        first_req_logged = True
    g.start_time = time.perf_counter()

@app.after_request
def log_request(response):
    duration = time.perf_counter() - g.get('start_time', time.perf_counter())
    if request.endpoint in ('api_predict', 'api_login', 'api_signup'):
        app.logger.info(f"{request.endpoint} took {duration:.3f}s")
    response.headers.update({
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    return response

#database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

#page routes
@app.route('/')
def index(): return render_template('index.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/login')
def login(): return render_template('login.html')

@app.route('/signup')
def signup(): return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# API: Auth
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username, password = data.get('username', '').strip(), data.get('password', '').strip()

    if not username or not password:
        return jsonify({'error': 'All fields are required.'}), 400

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode(), user.password.encode()):
        session.update({'username': user.username, 'id': user.id})
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid username or password.'}), 401

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    required = ['first_name', 'last_name', 'username', 'email', 'password', 'confirm_password']
    if not all(data.get(field, '').strip() for field in required):
        return jsonify({'error': 'All fields are required.'}), 400
    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match.'}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists.'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered.'}), 400

    hashed_pw = bcrypt.hashpw(data['password'].encode(), bcrypt.gensalt()).decode()
    new_user = User(
        first_name=data['first_name'], last_name=data['last_name'],
        username=data['username'], email=data['email'], password=hashed_pw
    )
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True})
    except:
        db.session.rollback()
        return jsonify({'error': 'Error creating account.'}), 500

#prediction page + history ---
@app.route('/predict')
def predict():
    league = request.args.get('league', '2024_2025')
    df = load_and_filter_data()
    teams = sorted(pd.unique(df[df['Season'] == league][['HomeTeam', 'AwayTeam']].values.ravel()))
    
    history = []
    if session.get('id'):
        records = Prediction.query.filter_by(user_id=session['id']).order_by(Prediction.timestamp).all()
        for p in records:
            ts = p.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            history.append(f"[{ts}] > {p.home_team} vs {p.away_team} → {p.result}")

    return render_template('predict.html', teams=teams, selected_league=league, history_entries=history)

#API: make prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    home, away, league = data.get('home_team'), data.get('away_team'), data.get('league', '2024_2025')

    if home == away:
        return jsonify({'error': 'Home and away teams must be different'}), 400
    if league != '2024_2025':
        return jsonify({'error': 'Prediction for this league is not available. Coming soon.'}), 400

    result, _ = predict_match(home, away)
    timestamp = datetime.utcnow()

    if session.get('id'):
        db.session.add(Prediction(user_id=session['id'], home_team=home, away_team=away, result=result, timestamp=timestamp))
        db.session.commit()

    return jsonify({
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'home_team': home, 'away_team': away, 'result': result
    })

#API: get teams for selected league
@app.route('/api/teams')
def api_teams():
    league = request.args.get('league', '2024_2025')
    df = load_and_filter_data()
    df_league = df[df['Season'] == league]
    if df_league.empty:
        return jsonify({'error': 'No teams found for selected league'}), 404
    teams = sorted(pd.unique(df_league[['HomeTeam', 'AwayTeam']].values.ravel()))
    return jsonify({'teams': teams})

#launch app
if __name__ == '__main__':
    app.run(debug=True)
