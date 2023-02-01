from OLD.ETL_data_stations import retrieveDataframe
from typing import List
import numpy as np
import pandas as pd

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

def getNumberOfInterventions(export_name, list_of_trainseries):
    df = pd.read_csv(export_name, sep=";")
    df = keepTrainseries(df, list_of_trainseries)
    # vergelijk prl_rijw_eerst_rijweg_van, prl_rijw_eerst_rijweg_naar, prl_rijw_eerst_rijweg_dwangnr, MET prl_rijw_laatst_rijweg_van, prl_rijw_laatst_rijweg_naar prl_rijw_laatst_rijweg_dwangnr

    df['prl_rijw_eerst_rijweg_dwangnr'] = df['prl_rijw_eerst_rijweg_dwangnr'].apply(pd.to_numeric, errors='coerce')
    df['prl_rijw_laatst_rijweg_dwangnr'] = df['prl_rijw_laatst_rijweg_dwangnr'].apply(pd.to_numeric, errors='coerce')

    df[['prl_rijw_eerst_rijweg_van', 'prl_rijw_laatst_rijweg_van', 'prl_rijw_eerst_rijweg_naar', 'prl_rijw_laatst_rijweg_naar']] = df[['prl_rijw_eerst_rijweg_van', 'prl_rijw_laatst_rijweg_van', 'prl_rijw_eerst_rijweg_naar', 'prl_rijw_laatst_rijweg_naar']].fillna("Emp")
    df[['prl_rijw_eerst_rijweg_dwangnr', 'prl_rijw_laatst_rijweg_dwangnr']] = df[['prl_rijw_eerst_rijweg_dwangnr', 'prl_rijw_laatst_rijweg_dwangnr']].fillna(0)

    df['intervened'] = np.where( (df['prl_rijw_eerst_rijweg_van'] == df['prl_rijw_laatst_rijweg_van']) &
                                 (df['prl_rijw_eerst_rijweg_naar'] == df['prl_rijw_laatst_rijweg_naar']) &
                                 (df['prl_rijw_eerst_rijweg_dwangnr'] == df['prl_rijw_laatst_rijweg_dwangnr']), False, True )
    pd.set_option('display.max_columns', None)
    intv = (df['intervened'] == True).sum()
    no_intv = (df['intervened'] == False).sum()
    return ((100 /(no_intv + intv)) * intv).round(2)

def getInfo(export_name):
    list_of_trainseries = ["500E","500O","600E","600O","700E","700O",'1800E',"1800O","6200E","6200O","8100E",'8100O']
    df = retrieveDataframe(export_name, True)
    df = keepTrainseries(df, list_of_trainseries)
    gb = df.groupby(['date'])
    print("Amount of days:", len(gb.groups))
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    print("Different group lengths", getAllGroupLengths(grouped_by_date))
    total_lengths = [len(x) for x in grouped_by_date]
    most_common_length = max(set(total_lengths), key=total_lengths.count)
    print("Largest group with length:", most_common_length)
    print("Percentage of interventions", getNumberOfInterventions(export_name, list_of_trainseries))

    grouped_by_date.sort(key=lambda x: x.date.iloc[0], reverse=False)
    d = {}  # {500O: {50:2, 51:5, 44:20}}
    for serie in list_of_trainseries:
        d_amount = {}
        for df_g in grouped_by_date:
            size = len(df_g.loc[df_g['basic_treinnr_treinserie'] == serie])
            d_amount[size] = d_amount.get(size, 0) + 1
            print(serie, df_g['date'].iloc[0], len(df_g.loc[df_g['basic_treinnr_treinserie'] == serie]))
        d[serie] = d_amount

    d_res = {}
    for k, v in d.items():
        v = {k: v1 for k, v1 in sorted(v.items(), key=lambda item: item[1], reverse=True)}
        print(k, v)
        d_res[k] = list(v.keys())[0]

#getInfo('../Data/asn_bl_hgv_mp_2019_total.csv')