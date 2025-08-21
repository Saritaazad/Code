# Native libraries
import os
import math

# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Algorithms
from minisom import MiniSom
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import geopandas as gpd
from sklearn.neighbors import KDTree
from matplotlib import font_manager

# Font settings
font_manager.findfont("Arial") 
mpl.rcParams.update({"font.family": "Arial"})
mpl.rcParams.update({"font.size": "10"})

# Load data
df = pd.read_csv('TimeSeriesDataHP.csv', header=None)
data = df.to_numpy()
xLong = list(df.iloc[:, 0])  # Longitude
yLat = list(df.iloc[:, 1])   # Latitude

# Plotting shapefile
shp = gpd.read_file('NWH/4-17-2018-899072.shp')
shp.plot(figsize=(8, 12))

loc_plot = pd.DataFrame({"Longitude": xLong, "Latitude": yLat})
sns.scatterplot(x="Longitude", y="Latitude", data=loc_plot, style=True, legend=False, markers="^")
plt.show()

# All DATA
mySeries = data

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(mySeries, test_size=0.10, random_state=42)

X_train = mySeries

#som_x = som_y = 6
#som_x = som_y = 7
som_x = som_y = 8

som = MiniSom(som_x, som_y, len(mySeries[0]), sigma=0.3, learning_rate=0.1)

som.random_weights_init(X_train)
som.train(X_train, 1500)


def is_valid_number(s):
    """
    Check if the given string is a valid number.
    
    Args:
        s (str): The string to check.
    
    Returns:
        bool: True if the string is a valid number, False otherwise.
    """
    try:
        print('The value of s is =', str(s))
        float(s)
        return not math.isnan(s)
    except ValueError:
        return False

def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x, som_y, figsize=(10, 10))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x, y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series, c="gray", alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]), axis=0), c="red")
            cluster_number = x * som_y + y + 1
            #axs[cluster].set_title(f"Cluster {cluster_number}")
    plt.show()

win_map = som.win_map(X_train)

plot_som_series_averaged_center(som_x, som_y, win_map)

# countOfNonZeroClusters = 0
# cluster_c = []
# cluster_n = []
# for x in range(som_x):
#     for y in range(som_y):
#         cluster = (x, y)
#         if cluster in win_map.keys():
#             if(len(win_map[cluster])>0):
#                 cluster_c.append(len(win_map[cluster]))
#                 countOfNonZeroClusters = countOfNonZeroClusters+1
#         else:
#             cluster_c.append(0)
#         cluster_number = x * som_y + y + 1
#         cluster_n.append(f" {cluster_number}")

countOfNonZeroClusters = 0
cluster_c = []
cluster_n = []
plt.bar(cluster_n, cluster_c)
x_count = 0

for x in range(som_x):
    for y in range(som_y):
        cluster = (x, y)
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^cluster^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # print(cluster)

        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        if cluster in win_map.keys():
            if(len(win_map[cluster])>0):
                print(len(win_map[cluster]))
                cluster_c.append(len(win_map[cluster]))
                countOfNonZeroClusters = countOfNonZeroClusters+1
                #cluster_number = x * som_y + y + 1 # some weird logic here? cluster_number = x * som_y + y + 1 
                x_count = x_count+1
                cluster_number = x_count
                print('custer_number=',str(cluster_number))
                cluster_n.append(f" {cluster_number}")
        # else:
        #     cluster_c.append(0)
        # cluster_number = x * som_y + y + 1
        # cluster_n.append(f" {cluster_number}")
print('----------------------------------------------')
print('length of cluster_c=', str(len(cluster_c)))
print('length of cluster_n=', str(len(cluster_n)))
print('----------------------------------------------')
plt.figure(figsize=(25,5))
plt.bar(cluster_n, cluster_c, color='skyblue')
plt.show()

"""
# Another way to create the BAR CHART
dataset = pd.DataFrame({'cluster_c': cluster_c, 'cluster_n': cluster_n}, columns=['cluster_c', 'cluster_n'])
y_pos = np.arange(len(cluster_n))

plt.barh(y_pos, cluster_c, align='center', alpha=1)
plt.yticks(y_pos, cluster_n, size=8)
plt.xticks([0, 5, 10, 15, 20], size=8)
plt.xlabel('Number of Grids', size=15)
plt.ylabel('Group ID', size=15)
plt.title('Distribution of Number of Grid Locations across the Homogeneous Groups', size=15)
plt.show()

"""

for series in mySeries[:5]:
    print(som.winner(series))

c = [
    'dodgerblue', 'salmon', 'green', 'yellow', 'darkblue', 'lightgreen',
    'darkred', 'pink', 'darkgreen', 'grey', 'brown', 'purple', 'black',
    'lightpink', 'gold', 'darkblue', 'olive', 'orange', 'cyan', 'tan',
    'lawngreen', 'rosybrown', 'coral', 'firebrick', 'chocolate', 'darkorange',
    'khaki', 'darkkhaki', 'greenyellow', 'darkseagreen', 'aquamarine',
    'cadetblue', 'powderblue', 'slateblue', 'blueviolet', 'plum'
]

w = []
selNodesWithPrecipData = []
labelsTest = []
dfC = pd.DataFrame(columns=['Longitude', 'Latitude', 'labels', 'TotalPrec','Rainfall'])

X_train = mySeries

locPlot2DA = X_train
forCorrMat = []
for idx in range(len(X_train)):
    winner_node = som.winner(X_train[idx])
    w.append(winner_node)
    selNodesWithPrecipData.append(locPlot2DA[idx])
    temp = locPlot2DA[idx]
    temp = np.append(temp, winner_node[0] * som_y + winner_node[1] + 1)
    forCorrMat.append(temp)

    total = 0.0
    for k in range(2, len(locPlot2DA[idx])):
        total += float(locPlot2DA[idx][k])
    
    Rainfall = []
    for k in range(2, len(locPlot2DA[idx])):
        Rainfall.append(float(locPlot2DA[idx][k]))

    labelsTest.append(winner_node[0] * som_y + winner_node[1] + 1)
    row = {
        "Longitude": locPlot2DA[idx][0],
        "Latitude": locPlot2DA[idx][1],
        "labels": winner_node[0] * som_y + winner_node[1] + 1,
        "TotalPrec": total,
        "Rainfall": Rainfall
    }
    dfC = dfC.append(row, ignore_index=True)
print('selNodesWithPrecipData',len(selNodesWithPrecipData))
print('w=',len(w))
print('w[0]=',len(w[0]))
shp.plot(figsize=(10, 14))
markerList = []

for h in range(countOfNonZeroClusters):
    str1 = str(h)
    markerList.append(str1)

sns.scatterplot(x="Longitude", y="Latitude", data=dfC, hue="labels", palette="bright", legend=False, s=100, style="labels")
plt.title("Patterns with for SOM", size=15)

plt.xlabel('Longitude', size=15)
plt.ylabel('Latitude', size=15)

# Adjust the values of 'fontsize' accordingly
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()

# Convert coordinates to radians
A1 = dfC.to_numpy()
raddfC = pd.DataFrame(columns=['Longitude', 'Latitude', 'labels'])
for idx in range(len(A1)):
    rad_Long = np.deg2rad(A1[idx][0])
    rad_Lat = np.deg2rad(A1[idx][1])
    row = {
        "Longitude": rad_Long,
        "Latitude": rad_Lat,
        "labels": A1[idx][2]
    }
    raddfC = raddfC.append(row, ignore_index=True)

for idx in range(len(X_test)):
    X_test[idx][0] = np.deg2rad(X_test[idx][0])
    X_test[idx][1] = np.deg2rad(X_test[idx][1])

kd = KDTree(raddfC[["Longitude", "Latitude"]].values, metric='euclidean')
CorrectPredictions = 0
WrongPredictions = 0

###################To check the kd tree for prediction
long = 79.09
lat = 29.6
rad_Long = np.deg2rad(long)
rad_Lat = np.deg2rad(lat)
    
# Find the nearest latitude and longitude from the grid
temp = [[rad_Long, rad_Lat]]
    
distances, indices = kd.query(temp, k=1)
    
nearest_index = indices[0][0]

clusId = int(A1[indices[0][0]][2])
print('lat=',str(lat), ' long=', str(long))
print('nearest_index')
print(nearest_index)
print('clusId==', clusId)




########################################################


for i in range(len(X_test)):
    temp = [[X_test[i][0], X_test[i][1]]]
    
    distances, indices = kd.query(temp, k=1)
    
    clusId = int(A1[indices[0][0]][2])
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    # print('indices')
    # print(indices)
    # print('clusId=', clusId)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    for k in range(len(A1)):
        #print('A1[k][2]',str(A1[k][2]))
        if(A1[k][2]==clusId):
            for p in range(len(A1[k][4])):
                if abs(float(A1[k][4][p]) - float(X_test[i][p])) <= 4:
                    CorrectPredictions += 1
                else:
                    WrongPredictions += 1
            break
    # for m in range(2, len(X_test[i]) - 2):
    #     if abs(float(X_test[i][m]) - float(selNodesWithPrecipData[clusId][m])) <= 1:
    #         CorrectPredictions += 1
    #     else:
    #         WrongPredictions += 1

print('The total length of data is', len(X_test))
print('CORRECT PREDICTIONS', CorrectPredictions)
print('WRONG PREDICTIONS', WrongPredictions)

################# for showing centriods START ##########################
# print('--------Labels--------')

# column_names = dfC.columns
# print('Column names in dfC')
# print(column_names)
file_path_cenroid = "centroid64.csv"

ArryLongList=[]
ArryLatList=[]

strLabel = 'labels'
strLong = 'Longitude'
strLat = 'Latitude'
strTotPrec = 'TotalPrec'

A = dfC[strLabel]
arryLong = dfC[strLong]
arryLat = dfC[strLat]
arryTotalPrec = dfC[strTotPrec]

# print(A)
# print('----------------')
# print(A[5])


for i in range(1,64):
    NumberOfGridsInCluster = 0
    clusterLong = []
    clusterLat = []
    clusterTotPrec = []
    for j in range(len(A)):
        if(i == int(A[j])):
            clusterLong.append(arryLong[j])
            clusterLat.append(arryLat[j])
            clusterTotPrec.append(arryTotalPrec[j])
            NumberOfGridsInCluster=NumberOfGridsInCluster+1

    centroid_longitude = np.mean(clusterLong)
    centroid_latitude = np.mean(clusterLat)

    #########to get The KD nearest neighbour ################
    if (is_valid_number(centroid_longitude)&is_valid_number(centroid_latitude)):
        rad_Long = np.deg2rad(centroid_longitude)
        rad_Lat = np.deg2rad(centroid_latitude)
        # print('centroid_longitude=',str(centroid_longitude))
        # print('centroid_latitude=',str(centroid_latitude))
        temp = [[rad_Long, rad_Lat]]
        distances, indices = kd.query(temp, k=1)
        # print('distance = ',str(distances))
        # print('indices = ', indices)

        longV = float(A1[indices[0][0]][0])
        latV = float(A1[indices[0][0]][1])

        print('longV = ',str(longV))
        print('latV = ', str(latV))

 #########################################################

        ArryLongList.append(longV)
        ArryLatList.append(latV)

with open(file_path_cenroid, "w") as file:
    for i in range(len(ArryLatList)):
        long1, lat1 = ArryLongList[i], ArryLatList[i]
        file.write(f"{long1},{lat1}\n")



# Function to generate predictions for a given latitude and longitude
def generate_prediction(lat, long):
    # Convert latitude and longitude to radians
    rad_Long = np.deg2rad(long)
    rad_Lat = np.deg2rad(lat)
    
    # Find the nearest latitude and longitude from the grid
    temp = [[rad_Long, rad_Lat]]
    
    distances, indices = kd.query(temp, k=1)
    
    nearest_index = indices[0][0]

    clusId = int(A1[indices[0][0]][2])
    print('lat=',str(lat), ' long=', str(long))
    print('nearest_index')
    print(nearest_index)
    print('clusId==', clusId)
    
    
    # Get the rainfall data from the nearest grid location
    rainfall_data = locPlot2DA[nearest_index][2:]
    
    # Return the rainfall data as the prediction
    return rainfall_data

# Generate predictions and write to CSV
#test_locations = [(31.45, 76.7), (76.75, 31.35)]  # Example locations

test_locations = [(74.77, 34.21), (74.7, 34.6), (76.35, 34.19), (77.6, 34.16), (77.57, 34.15), (77.01, 31.89), (76.7, 31.7), (74.79, 34.08), (74.14, 33.6), (74.71, 34.4), (75.31, 34.01), (77.53, 34.31), (80.36, 30.08), (79.09, 29.6)
                    , (78.68, 30.57), (78.62, 30.46), (78.775, 30.147)]  # Example locations

for lat, long in test_locations:
    # Generate prediction
    prediction = generate_prediction(long, lat)
    
    # Write prediction to CSV file
    filename = f"pred_{lat}_{long}.csv"
    prediction_df = pd.DataFrame({"Prediction": prediction})
    prediction_df.to_csv(filename, index=False)
    
    print(f"Prediction for latitude {lat} and longitude {long} has been saved to {filename}")






