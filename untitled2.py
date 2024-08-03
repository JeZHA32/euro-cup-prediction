import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Load the data
data = pd.read_csv('21.csv')

# Print column names to inspect
print("Available columns:")
print(data.columns)

# Step 2: Prepare the data
features = [
    'home_team', 'away_team', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target', 'home_possession',
    'away_possession', 'home_passes', 'away_passes', 'home_pass_success',
    'away_pass_success', 'home_fouls', 'away_fouls', 'home_yellow_cards',
    'away_yellow_cards', 'home_red_cards', 'away_red_cards', 'home_offsides',
    'away_offsides', 'home_corners', 'away_corners'
]
all_targets = ['home_score', 'away_score', 'home_shots', 'away_shots',
               'home_shots_on_target', 'away_shots_on_target','home_corners', 'away_corners']

# Filter targets to only include those present in the dataset
targets = [target for target in all_targets if target in data.columns]

print("\nUsing targets:", targets)

X = data[features]
y = data[targets]

# Step 3: Feature engineering function
def create_features(X):
    X = X.copy()
    X['total_shots'] = X['home_shots'] + X['away_shots']
    X['shot_accuracy_diff'] = (X['home_shots_on_target'] / X['home_shots']) - (X['away_shots_on_target'] / X['away_shots'])
    X['possession_diff'] = X['home_possession'] - X['away_possession']
    X['pass_accuracy_diff'] = X['home_pass_success'] - X['away_pass_success']
    X['foul_diff'] = X['home_fouls'] - X['away_fouls']
    return X

X = create_features(X)

# Update features list to include new features
features = X.columns.tolist()

# Step 4: Create a preprocessing pipeline with one-hot encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), [
            'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
            'home_possession', 'away_possession', 'home_passes', 'away_passes',
            'home_pass_success', 'away_pass_success', 'home_fouls', 'away_fouls',
            'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards',
            'home_offsides', 'away_offsides', 'home_corners', 'away_corners',
            'total_shots', 'shot_accuracy_diff', 'possession_diff', 'pass_accuracy_diff', 'foul_diff'
        ]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team'])
    ])

# Create a pipeline that combines preprocessing and the XGBoost model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Step 5: Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Step 6: Perform hyperparameter tuning with GridSearchCV
param_grid = {
'regressor__n_estimators': [100,200],
'regressor__max_depth': [1],
'regressor__learning_rate': [0.03],
'regressor__subsample': [0.4,0.7],
'regressor__colsample_bytree': [0.7,0.9,1],
'regressor__min_child_weight': [1,3,5,7,9],
'regressor__gamma': [0.1,0.3]
}

grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R-squared score: {grid_search.best_score_}")

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 7: Make predictions
def predict_match(home_team, away_team):
    # Find average features for the given home and away team
    home_team_data = data[data['home_team'] == home_team].mean().fillna(0)
    away_team_data = data[data['away_team'] == away_team].mean().fillna(0)

    # Combine the average features into a single input
    input_data = pd.DataFrame([[
        home_team, away_team,
        home_team_data['home_shots'], away_team_data['away_shots'],
        home_team_data['home_shots_on_target'], away_team_data['away_shots_on_target'],
        home_team_data['home_possession'], away_team_data['away_possession'],
        home_team_data['home_passes'], away_team_data['away_passes'],
        home_team_data['home_pass_success'], away_team_data['away_pass_success'],
        home_team_data['home_fouls'], away_team_data['away_fouls'],
        home_team_data['home_yellow_cards'], away_team_data['away_yellow_cards'],
        home_team_data['home_red_cards'], away_team_data['away_red_cards'],
        home_team_data['home_offsides'], away_team_data['away_offsides'],
        home_team_data['home_corners'], away_team_data['away_corners'],
        home_team_data['home_shots'] + away_team_data['away_shots'],
        (home_team_data['home_shots_on_target'] / home_team_data['home_shots']) - (away_team_data['away_shots_on_target'] / away_team_data['away_shots']),
        home_team_data['home_possession'] - away_team_data['away_possession'],
        home_team_data['home_pass_success'] - away_team_data['away_pass_success'],
        home_team_data['home_fouls'] - away_team_data['away_fouls']
    ]], columns=features)

    # Make prediction
    prediction = best_model.predict(input_data)

    return prediction[0]

# Example usage
home_country = "Austria"
away_country = "Turkey"
prediction = predict_match(home_country, away_country)

print(f"\nPredicted match outcome for {home_country} vs {away_country}:")
for target, value in zip(targets, prediction):
    print(f"{target}: {value:.2f}")
