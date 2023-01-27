import numpy as np
import datetime
import time

from createSCM import createCGWithPC, createCGWithFCI, createCGWithDirectLiNGAM
from ETL_data_day import TRN_matrix_to_delay_matrix_columns_pair, dfToTrainRides
from Load_transform_df import retrieveDataframe
from createSuperGraph import get_CG_and_superGraph
from createBackground import variableNamesToNumber
from createSCM import createCGWithGES
from FAS import FAS, createIDTRNDict

def main():
    print("extracting file")
    export_name = 'Data/6100_jan_nov_2022_2.csv' #'Data/2019-03-01_2019-05-31.csv'
    list_of_trainseries = ['6100']

    # extract dataframe and impute missing values
    df = retrieveDataframe(export_name, True, list_of_trainseries)
    # change the dataframe to trainRideNodes
    dataset_with_classes = dfToTrainRides(df)

    print("extracting file done")

    print("translating dataset to 2d array for algo")
    # have a smaller dataset for testing purposes
    smaller_dataset = dataset_with_classes[:,:10] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
    # get the schedule
    sched_with_classes = smaller_dataset[0]
    # translate the TrainRideNodes to delays
    res_dict = TRN_matrix_to_delay_matrix_columns_pair(smaller_dataset)
    delays_to_feed_to_algo, column_names = res_dict['delay_matrix'], res_dict['column_names']

    # create a background and its schedule (background for Pc or FCI, cg_sched for GES)
    bk, cg_sched = get_CG_and_superGraph(sched_with_classes, 'Results/sched.png') #get_CG_and_background(smaller_dataset, 'Results/sched.png')

    # independence test methods for Pc or FCI
    method = 'mv_fisherz' #'fisherz'
    trn_name_id_dict, id_trn_name_dict = variableNamesToNumber(sched_with_classes)
    #create a Causal Graph
    id_trn_dict = createIDTRNDict(sched_with_classes)
    createCGWithGES(delays_to_feed_to_algo, 'Results/6100_jan_nov_with_backg.png', 'local_score_BIC', column_names)
    #FAS(method, delays_to_feed_to_algo, 'Results/6100_jan_nov_with_backg.png', id_trn_dict, id_trn_name_dict, column_names, bk)

    # createCGWithFCI(method, delays_to_feed_to_algo, 'Results/6100_jan_nov_wo_backgr.png', column_names)

main()