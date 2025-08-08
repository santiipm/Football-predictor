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
def generate_new_features(data,matches=5):
    try:
        data=data.sort_values(by=['team','date']).reset_index(drop=True)

        data['gf_avg']=0.0
        data['ga_avg']=0.0
        data['poss_avg']=0.0
        data['points']=data['result'].apply(lambda x: 3 if x=='W' else 1 if x=='D' else 0)
        data['points_total']=0.0
        data['xg_avg']=0.0
        data['xga_avg']=0.0

        for i,row in data.iterrows():
            team_matches=data.loc[(data['team']==row['team'])&(data['date']<row['date'])].tail(matches)
            if len(team_matches)==matches:
                data.loc[i,'gf_avg']=team_matches['gf'].mean()
                data.loc[i,'ga_avg']=team_matches['ga'].mean()
                data.loc[i, 'poss_avg'] = team_matches['poss'].mean()
                data.loc[i, 'points_total'] = team_matches['points'].sum()
                data.loc[i, 'xg_avg'] = team_matches['xg'].mean()
                data.loc[i, 'xga_avg'] = team_matches['xga'].mean()

        data.drop(columns=['points'])

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
        processed_data=generate_new_features(cleaned_data)
        print("Data with new features generated successfully. First 5 lines: ")
        print(cleaned_data.head())
        print("Dataframe information after the features: ")
        print(cleaned_data.info())