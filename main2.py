import matplotlib.pyplot as plt
from ETL_data_day import TRN_matrix_to_delay_matrix_columns_pair, dfToTrainRides
from Load_transform_df import retrieveDataframe
from createSuperGraph import DomainKnowledge, Graph_type
from OLD.createBackground import variableNamesToNumber
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from FAS import FAS_method
from graphToInput import NN_samples
from Utils import gg2txt
import math
import numpy as np

def main():
    print("extracting file")
    export_name =  'Data/2019-03-01_2019-05-31_original.csv' #'Data/Ut_2022-01-01_2022-12-10_2.csv' #'Data/6100_jan_nov_2022_2.csv'
    list_of_trainseries= ['500E', '500O', '600E', '600O', '700E','700O','1800E','1800O''6200E','6200O','8100E','8100O','9000E','9000O','12600E',
                           #'32200E''32200O','32300E','32300O',
                           '76200O','78100E','78100O','79000E','79000O'
                            #,'80000E','80000O','80200E','80200O','89200E','89200O','93200E','93200O'
                           #, '301800O','332200E','406200O'
                           ]

    #list_of_trainseries_Ut = ['104VB', '120NB', '120VB', '220NB', '220VB', '402NB', '402VB', '420NB', '420VB', '500E', '500O', '600E', '600O', '800E', '800O', '1400E', '1400O', '1700E', '1700O', '2000E', '2000O', '2800E', '2800O', '2900E', '2900O', '3000E', '3000O', '3100E', '3100O', '3500E', '3500O', '3700E', '3700O', '3900E', '3900O', '4900E', '4900O', '5500E', '5500O', '5600E', '5600O', '5700E', '5700O', '6000E', '6000O', '6900E', '6900O', '7300E', '7300O', '7400E', '7400O', '8800E', '8800O']

    # extract dataframe and impute missing values
    df, sched = retrieveDataframe(export_name, True, list_of_trainseries)
    print("done extracting", len(df))
    # change the dataframe to trainRideNodes
    dataset_with_classes = dfToTrainRides(df)
    sched_with_classes = dfToTrainRides(sched)[0]
    print("extracting file done")

    print("translating dataset to 2d array for algo")
    # have a smaller dataset for testing purposes
    smaller_dataset = dataset_with_classes[:,:] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
    sched_with_classes = sched_with_classes[:]
    print("Amount of variables: ", len(dataset_with_classes[0]))
    # get the schedule

    # translate the TrainRideNodes to delays
    column_names = np.array(list(map(lambda x: x.getID(), sched_with_classes)))
    res_dict = TRN_matrix_to_delay_matrix_columns_pair(smaller_dataset)
    delays_to_feed_to_algo = res_dict['delay_matrix']#, res_dict['column_names']

    # create a background and its schedule (background for Pc or FCI, cg_sched for GES)
    dk = DomainKnowledge(sched_with_classes, 'Results/sched.png', Graph_type.MINIMAL)
    bk, cg_sched = dk.get_CG_and_superGraph() #get_CG_and_background(smaller_dataset, 'Results/sched.png')

    # independence test methods for Pc or FCI
    method = 'mv_fisherz' #'fisherz'
    trn_name_id_dict, id_trn_name_dict = variableNamesToNumber(sched_with_classes)
    #create a Causal Graph
    fas_method = FAS_method(method, delays_to_feed_to_algo, 'Results/6100_jan_nov_with_backg_FAS.png', sched_with_classes, id_trn_name_dict, column_names, bk)
    print(delays_to_feed_to_algo)
    gg_fas = fas_method.fas_with_background()
    gg2txt(gg_fas, "6100_FAS2.txt", id_trn_name_dict)
    gg = txt2generalgraph("6100_FAS2.txt")

    events = [("Bl", "8100O"), ("Bl", "8100E"), ("Hgv", "8100O"), ("Hgv", "8100E"), ("Asn", "500O"),("Asn", "700O"), ("Asn", "8100O"), ("Mp", "500E"), ("Mp", "700E"), ("Mp", "8100E")]
    data_collection = []
    for event in events:
        sample_changer = NN_samples(gg, fas_method, df)
        sample_changer.gg_to_nn_input(*event)
        sample_changer.findDelaysFromData()
        data = sample_changer.filterPrimaryDelay()
        data_collection.append(data)

    for i, data_sample in enumerate(data_collection):
        print(events[i])
        delays = list(filter(lambda x: x<=900 and x >= -300, data_sample))
        print(data_sample)
        bins = math.ceil((max(delays) - min(delays)) / 60)
        plt.hist(delays, bins=bins)
        plt.show()


main()