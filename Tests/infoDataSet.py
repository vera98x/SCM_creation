from ETL_data import retrieveDataframe
from typing import List

def getAllGroupLengths(grouped_by_date):
    total_lengths = [len(x) for x in grouped_by_date]
    d = {}
    for x in total_lengths:
        d[x] = d.get(x, 0) + 1
    return d

def amountIntervened():
    return

def keepTrainseries(df_input, act_val : List[str]):
    df_input = df_input.loc[df_input['basic_treinnr_treinserie'].isin(act_val)]
    return df_input

def getInfo(export_name):
    list_of_trainseries = ["500E","500O","600E","600O","700E","700O",'1800E',"1800O","6200E","6200O","8100E",'8100O']
    df = retrieveDataframe(export_name, None)
    df = keepTrainseries(df, list_of_trainseries)
    gb = df.groupby(['date'])
    print("Amount of days:", len(gb.groups))
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    print("Different group lengths", getAllGroupLengths(grouped_by_date))
    total_lengths = [len(x) for x in grouped_by_date]
    most_common_length = max(set(total_lengths), key=total_lengths.count)
    print("Largest group with length:", most_common_length)

getInfo('../Data/2019-03-01_2019-05-31.csv')