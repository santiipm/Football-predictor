import pandas as pd

def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path)
        cols_to_drop= ['comp', 'round', 'day' ,'attendance', 'captain', 'formation', 'opp formation', 'referee',
                       'match report', 'notes']
        data=data.drop(cols_to_drop, errors='ignore')
        data.columns=[c.replace(' ', '_').lower() for c in data.columns]
        data['date'] = pd.to_datetime(data['date'])
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
if __name__ == '__main__':
    file_path='data/matches_full.csv'
    cleaned_data = load_and_clean_data(file_path)

    if cleaned_data is not None:
        print("Data loaded and cleaned successfully. First 5 lines: ")
        print(cleaned_data.head())
        print("Dataframe information after the cleaning: ")
        print(cleaned_data.info())