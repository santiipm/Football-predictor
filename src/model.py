from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from data_processing import load_and_clean_data,generate_new_features

def prepare_data(file_path):
    clean_data = load_and_clean_data(file_path)
    if clean_data is None:
        return None,None
    processed_data = generate_new_features(clean_data,5)
    if processed_data is None:
        return None, None

    features=['gf_avg','ga_avg','xg_avg','xga_avg','points_total','poss_avg']
    X=processed_data[features]
    y=processed_data['result']
    return X,y
if __name__=='__main__':
    file_path='data/matches_full.csv'
    X,y=prepare_data(file_path)
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('X and Y have been divided. Size of the training group:')
        print(len(X_train))
        print('\nSize of the testing group:')
        print(len(X_test))

        model=RandomForestClassifier(n_estimators=100, random_state=42,min_samples_split=10,min_samples_leaf=5)
        model.fit(X_train,y_train)
        print("model trained successfully")
