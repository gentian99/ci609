# ci609  
Final Year Project

# Football Match Outcome Prediction Web Application

This is a web-based football match outcome prediction system developed using Python, Flask, and TensorFlow. The application allows users to select two football teams and receive a predicted outcome based on historical match data and engineered performance metrics. It incorporates a hybrid machine learning model combining tabular features and sequence data.

## Overview

- Built using Flask for the web backend and TensorFlow for model inference.
- Implements a hybrid neural network trained on Premier League data (2022–2025).
- Provides user authentication and result history functionality.
- Supports Premier League team selection with placeholder support for additional leagues.

## Prerequisites

Before setting up the project, ensure the following are available:

- [pyenv](https://github.com/pyenv/pyenv) – for managing Python versions
- Python 3.10.13 (installed via `pyenv`)
- pip (Python package installer)
- Internet connection for downloading dependencies and datasets

**Important:**  
TensorFlow and several other scientific libraries do **not currently support Python 3.11, 3.12, or 3.13**.  
Using Python 3.13 will result in install errors such as:


## Installation Instructions

### 1. Clone the Repository

Open a terminal and run the following commands to clone the repository and navigate into the project directory:

```bash
git clone https://github.com/gentian99/ci609.git
cd ci609/ci609-project
```

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment:

```bash
pyenv install 3.10.13               
pyenv local 3.10.13                   
rm -rf .venv                         
python -m venv .venv                
source .venv/bin/activate  
pip install --upgrade pip        
```

### 3. Install Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Train the Model

Run the model training script. This will:

- Load and preprocess historical match data.
- Train an ensemble of hybrid neural networks.
- Save the final model and preprocessing objects.

```bash
python train_model.py
```

This will generate:
- `football_hybrid_model_final.h5`
- `scaler.pkl`
- `feature_columns.pkl`
- Fold-specific model files for ensemble averaging.

### 5. Initialize the Database

Create the required database tables for user accounts and predictions:

```bash
python create_db.py
```

This will generate `site.db` in the `instance/` directory using SQLAlchemy.

### 6. Start the Flask Server

Run the application locally:

```bash
python app.py
```

Once the server is running, navigate to:

```
http://127.0.0.1:5000/
```

## Application Usage

- **Home Page**: Overview of the prediction system with background visuals and feature highlights.
- **About Page**: Describes the algorithm, responsible AI practices, and data privacy details.
- **Predict Page**:
  - Select a league and teams.
  - Run predictions.
  - View result and prediction history.
  - Save/export predictions when logged in.
- **Authentication**: Users can register and log in to save prediction results to their account.

## File Structure

```
├── app.py
├── train_model.py
├── prediction.py
├── preprocessing.py
├── create_db.py
├── main.py
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── login.html
│   ├── signup.html
│   └── predict.html
├── static/
│   ├── css/stylesheet.css
│   ├── js/script.js
│   └── videos/video1.mp4, video2.mp4
└── instance/
    └── site.db
```

## Notes

- The predictor uses historical data from [football-data.co.uk](https://www.football-data.co.uk/).
- Currently, predictions are limited to the Premier League 2024–2025 season, though dropdowns include additional leagues for future expansion.
- The model does not rely on betting data or odds and is intended for educational purposes only.

## License

This project was developed as part of an academic final-year Computer Science project. It is provided for research, demonstration, and educational purposes.
