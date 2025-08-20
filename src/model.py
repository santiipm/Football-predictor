import joblib
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from data_processing import load_and_clean_data,generate_new_features,generate_elo_ratings

def prepare_data(file_path):
    clean_data = load_and_clean_data(file_path)
    if clean_data is None:
        return None,None
    processed_data = generate_elo_ratings(clean_data)
    processed_data = generate_new_features(processed_data,5)
    if processed_data is None:
        return None, None

    features=['gf_avg','ga_avg','xg_avg','xga_avg','points_total','poss_avg','goal_diff_avg','xg_diff_avg','h2h_points',
              'elo_team','venue','opp_gf_avg','opp_ga_avg','opp_xg_avg','opp_xga_avg','opp_points_total','opp_poss_avg',
              'opp_goal_diff_avg','opp_xg_diff_avg','opp_h2h_points','elo_opp']

    X=processed_data[features]
    y=processed_data['result']

    return X,y


# ... [keep data processing] ...

if __name__ == '__main__':
    file_path = 'data/matches_full.csv'
    X, y = prepare_data(file_path)

    if X is not None and y is not None:
        time_split = TimeSeriesSplit(n_splits=5)

        # Optimized class weights based on your results
        class_weights = {'W': 1.0, 'D': 1.7, 'L': 1.0}

        # Hyperparameter grid for tuning
        param_dist = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.6, 0.8],
            'class_weight': [class_weights, 'balanced', None]
        }

        # Randomized Search with TimeSeriesSplit
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_dist,
            n_iter=30,  # Try 30 different combinations
            cv=time_split,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X, y)

        print("Best parameters:", random_search.best_params_)
        print("Best cross-validation score:", random_search.best_score_)

        # Test best model
        best_model = random_search.best_estimator_

        # Final evaluation with best model
        accuracies = []
        for train_index, test_index in time_split.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

        print(f"\nFinal average accuracy: {np.mean(accuracies):.3f}")
        print("Feature importance:")
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance_df.head(10))
        confusionmatrix = confusion_matrix(y_test, predictions,labels=best_model.classes_)
        conf_matrix_df = pd.DataFrame(confusionmatrix, index=best_model.classes_, columns=best_model.classes_)
        print(confusionmatrix)
        joblib.dump(best_model, 'laliga_predictor.pkl')
        print("Model saved")
