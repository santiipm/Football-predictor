import pandas as pd

def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path)
        cols_to_drop= ['comp', 'round', 'day' ,'attendance', 'captain', 'formation', 'opp formation', 'referee',
                       'match report', 'notes']
        data=data.drop(cols_to_drop, errors='ignore')
        data.columns=[c.replace(' ', '_').lower() for c in data.columns]
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(by=['date']).reset_index(drop=True)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_elo_ratings(data,k=20,initial_rating=1500):
    elo_ratings={}
    data['elo_team']=float('nan')
    data['elo_opp']=float('nan')

    for i,row in data.iterrows():
        team,opp=row['team'],row['opponent']

        if team not in elo_ratings:
            elo_ratings[team]=initial_rating
        if opp not in elo_ratings:
            elo_ratings[opp]=initial_rating

        r_team,r_opp=elo_ratings[team],elo_ratings[opp]

        e_team=1/(1+10**((r_opp-r_team)/400))
        e_opp=1-e_team

        if row['result']=='W':
            s_team,s_opp=1,0
        elif row['result']=='D':
            s_team,s_opp=0.5,0.5
        elif row['result']=='L':
            s_team,s_opp=0,1

        elo_ratings[team]=r_team+k*(s_team-e_team)
        elo_ratings[opp]=r_opp+k*(s_opp-e_opp)

        data.loc[i,'elo_team']=r_team
        data.loc[i,'elo_opp']=r_opp

    return data

def generate_new_features(data,matches=5):
    try:
        data['gf_avg']=float('nan')
        data['ga_avg']=float('nan')
        data['poss_avg']=float('nan')
        data['points']=data['result'].map({'W':3,'D':1,'L':0})
        data['points_total']=float('nan')
        data['xg_avg']=float('nan')
        data['xga_avg']=float('nan')
        data['h2h_points']=float('nan')
        data['team_position']=data.groupby(['date','season'])['points'].rank(ascending=False, method='min')

        for i,row in data.iterrows():
            team_matches=data.loc[(data['team']==row['team'])&(data['date']<row['date'])].tail(matches)
            if len(team_matches)>0:
                data.loc[i,'gf_avg']=team_matches['gf'].mean()
                data.loc[i,'ga_avg']=team_matches['ga'].mean()
                data.loc[i, 'poss_avg'] = team_matches['poss'].mean()
                data.loc[i, 'points_total'] = team_matches['points'].sum()
                data.loc[i, 'xg_avg'] = team_matches['xg'].mean()
                data.loc[i, 'xga_avg'] = team_matches['xga'].mean()
            h2h = (data.loc[(data['team'] == row['team'])&(data['opponent']==row['opponent'])&(data['date']<row['date'])]
                   .tail(matches))
            data.loc[i,'h2h_points']=h2h['points'].sum()
            data['h2h_winrate']=data['h2h_points']/(3*matches)

        data['venue']=data['venue'].map({'Home':1,'Away':0})
        data[['gf_avg','ga_avg','poss_avg','points_total','xg_avg','xga_avg']]=\
            (data[['gf_avg','ga_avg','poss_avg','points_total','xg_avg','xga_avg']].fillna(0))
        data['goal_diff_avg']=data['gf_avg']-data['ga_avg']
        data['xg_diff_avg']=data['xg_avg']-data['xga_avg']

        opp_data=data[['date','opponent','gf_avg','ga_avg','poss_avg','points_total','xg_avg','xga_avg','goal_diff_avg'
            ,'xg_diff_avg','h2h_points','h2h_winrate','team_position']]

        opp_data.columns=['date','team','opp_gf_avg','opp_ga_avg','opp_poss_avg','opp_points_total','opp_xg_avg',
                          'opp_xga_avg','opp_goal_diff_avg','opp_xg_diff_avg','opp_h2h_points','opp_h2h_winrate',
                          'opp_team_position']

        data=pd.merge(data,opp_data,left_on=['date','opponent'],right_on=['date','team'],how='left')

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
        processed_data = generate_elo_ratings(cleaned_data)
        processed_data=generate_new_features(processed_data,5)
        print("Data with new features generated successfully. First 5 lines: ")
        print(processed_data.head())
        print("Dataframe information after the features: ")
        print(processed_data.info())