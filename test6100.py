import pandas as pd
import datetime

def changeToD(df_complete):
    df_filter = df_complete[(df_complete["basic|drp_act"] == 'K_A') | (df_complete["basic|drp_act"] == 'A')
                            | (df_complete["basic|drp_act"] == 'K_V') | (df_complete["basic|drp_act"] == 'V')]

    # find the unique values of A or V: then Rs and Vs should be kept
    unique = df_filter.drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'date'], keep=False)

    # change the activity of "vertrek" to "doorkomst"
    df_complete["basic|drp_act"] = df_complete["basic|drp_act"].replace('K_V', 'D')
    df_complete["basic|drp_act"] = df_complete["basic|drp_act"].replace('V', 'D')

    #remove the A or K_A values
    df_complete = df_complete[~(df_complete["basic|drp_act"] == 'K_A') & ~(df_complete["basic|drp_act"] == 'A')]

    df_res = pd.concat([unique, df_complete]).drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'date' ], keep='first')

    df_res = df_res.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])

    df_res = df_res.reset_index(drop=True)

    return df_res

def makeDataUniform(df):
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    # get first dataframe as example to compare from
    example = grouped_by_date[0]
    example_date = example.iloc[0]['date']
    print(example_date)
    # loop through every other frame, compare the columns
    for day_index in range(len(grouped_by_date)):
        day = grouped_by_date[day_index]
        day_date = day.iloc[0]['date']

        diff = pd.concat([example, day]).drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'basic|drp_act'],
                                                         keep=False)
        # if a dataframe differs, print the data of the frame and show difference
        if (len(diff) != 0):
            #TODO: if length of dataframe is too small remove it anyway?

            # remove the extra activities
            remove_extra_activities = diff[~(diff["date"] == example_date)]
            df_res = pd.concat([day, remove_extra_activities]).drop_duplicates(
                subset=['basic|treinnr', 'basic|drp', 'date'],
                keep=False)

            # add missing values
            add_extra_activities = diff[(diff["date"] == example_date)]
            add_extra_activities['date'] = day_date

            # TODO: when creating the dataset, remove the basic plan and basic uitvoer
            #add_extra_activities['basic|plan'].apply(lambda dt: datetime.datetime.combine(day_date, dt.time()))
            #add_extra_activities['basic|uitvoer'].apply(lambda dt: datetime.datetime.combine(day_date, dt.time()))

            df_res = pd.concat([df_res, add_extra_activities])
            df_res = df_res.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
            df_res = df_res.reset_index(drop=True)

            grouped_by_date[day_index] = df_res

            df_new = pd.concat(grouped_by_date)

    return df_new


# split dataframes in column
export_name = 'Data/6100_jan_nov_2022.csv'
df = pd.read_csv(export_name, sep=";")
df = df[
    ['basic_treinnr_treinserie','basic|treinnr', 'basic|drp', 'basic|drp_act', 'basic|plan', 'basic|uitvoer']]
df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%d-%m-%Y %H:%M')
df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%d-%m-%Y %H:%M')
df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
df['date'] = pd.to_datetime(df['basic|plan']).dt.date


df = changeToD(df)
pd.set_option('display.max_columns', 5)
df = makeDataUniform(df)

df['basic|drp'] = df['basic|drp'].astype('string')
df['basic|treinnr'] = df['basic|treinnr'].astype('string')
df['basic|drp_act'] = df['basic|drp_act'].astype('string')

df['plan|time'] = pd.to_datetime(df['basic|plan']).dt.time
df['uitvoer|time'] = pd.to_datetime(df['basic|uitvoer']).dt.time

df['delay'] = df['basic|uitvoer'] - df['basic|plan']
df['delay'] = df['delay'].map(lambda x: x.total_seconds())

print(df)
