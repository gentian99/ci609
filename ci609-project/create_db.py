from app import app, db, Prediction

with app.app_context():
    db.create_all()
    print("Database tables created!")
