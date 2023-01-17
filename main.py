import numpy as np
import datetime
import time

from createscm import createCGWithPC, createCGWithFCI, createCGWithDirectLiNGAM
from ETL_data import getDataSetWith_TRN, class_dataset_to_delay_columns_pair
from createBackground import get_CG_and_background

print("extracting file")
export_name = 'Data/2019-03-01_2019-05-31.csv' #'Data/6100_jan_nov_2022.csv'
list_of_trainseries = ["500E", "500O", "600E", "600O", "700E", "700O", '1800E', "1800O", "6200E", "6200O", "8100E", '8100O']
stationNodes = getDataSetWith_TRN(export_name, True, list_of_trainseries)

#example for one station
dataset_with_classes = stationNodes[0].dataset_with_classes
print("extracting file done")

print("translating dataset to 2d array for algo")
smaller_dataset = dataset_with_classes[:] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
print(smaller_dataset[0])
print(smaller_dataset[1])
print(smaller_dataset[2])
delays_to_feed_to_algo, column_names = class_dataset_to_delay_columns_pair(smaller_dataset)
print(column_names)
print("Creating background knowledge")
start = time.time()
bk, cg_sched = get_CG_and_background(smaller_dataset, 'Results/sched.png')
end = time.time()
print("creating schedule took", end - start, "seconds")
# pdy = GraphUtils.to_pydot(cg_sched.G, labels=column_names )
# pdy.write_png("sched.png")
print("start with FCI and background")
start = time.time()
createCGWithFCI(delays_to_feed_to_algo, 'Results/6100_jan_nov_with_backg.png', column_names, bk)
end = time.time()
print()
print("creating SCM with background is done, it took" , end - start, "seconds")
print("start with FCI without background")
start = time.time()
createCGWithFCI(delays_to_feed_to_algo, 'Results/6100_jan_nov_wo_backgr.png', column_names)
end = time.time()
print("creating SCM without background is done, it took" , end - start, "seconds")

# for i,j in (compareSCM(fci_cg, fci_cg_none)):
#   print(i, "   \t", j)