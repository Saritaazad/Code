import csv
import pandas as pd

#df = pd.DataFrame()
globalList = []
AllGriddedLocations = []
def extract_data_to_list(string):
    print("extract_data_to_list")
    print(string)
    return string.split('_')


def extract_columns(file_path):
    #ol_name = ['Year', 'Month', 'Date' 'Longitude', 'Lattitude', 'Day', 'Month', 'Year', 'T2M', 'Humidity', 'TempMax', 'TempMin', 'WD10M', 'WS10M', 'PREP']
    df = pd.read_csv(file_path, header=0)
    print(df.shape)
    print(df.head())
    headerList = df.columns.tolist()
    print('Length of header list')
    print(len(headerList))


    list2 = []
    for column in df.columns:
        list1 = df[column].tolist()
        list2.append(list1)
    k = len(list2)
    print('Length of dataLists')
    print(k)
    for i in range(2, 3678):
    #for i in range(1, k):
        if (i%7==0):
            print('for i =='+str(i))
            #print('header list 1th value is')
            #print(headerList[i])
            hList = extract_data_to_list(headerList[i])
            

            Long = hList[-2]
            digits_list = [int(Lg) for Lg in str(Long)]
            print("Long===="+str(Long))
            print("digits_list===="+str(digits_list))
            if len(digits_list)>=4:
                strN = str(digits_list[0])+str(digits_list[1])+"."+str(digits_list[2])+str(digits_list[3])
                print("strN")
                print(strN)
                Long = strN
            elif len(digits_list) == 3:
                strN = str(digits_list[0])+str(digits_list[1])+"."+str(digits_list[2])
                print("strN")
                print(strN)
                Long = strN



            lat = hList[-1]
            digits_list = [int(Lg) for Lg in str(lat)]
            print("----lat===="+str(lat))
            print("digits_list===="+str(digits_list))
            if len(digits_list)>=4:
                strN = str(digits_list[0])+str(digits_list[1])+"."+str(digits_list[2])+str(digits_list[3])
                print("strN")
                print(strN)
                lat = strN
            elif len(digits_list) == 3:
                strN = str(digits_list[0])+str(digits_list[1])+"."+str(digits_list[2])
                print("strN")
                print(strN)
                lat = strN

            print("The long lat")
            print(Long)
            print(lat)
            lst = []
            lst.append(lat)
            lst.append(Long)
            for el in list2[i]:
                if (float(el)<-900):
                    #putting 0 if the rainfall data is not valid
                    lst.append(0)
                else:
                    lst.append(el)

            #print(lst)
            if(float(lat)<=80 and float(lat)>=75 and float(Long)<=34 and float(Long)>=30):
                globalList.append(lst)
            AllGriddedLocations.append(lst)
extract_columns("Daily_Data_42_Years_1981-2022.csv")

#print("----List Size----------------------------------")
#print(len(globalList))
#print("----globalList top 5----------------------------------")
#print(globalList)

# open a new CSV file for writing

with open('TimeSeriesDataHP.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # write each list to a new row in the CSV file
    for row in globalList:
        writer.writerow(row)


with open('TimeSeriesDataALL.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # write each list to a new row in the CSV file
    for row in AllGriddedLocations:
        writer.writerow(row)

#run on command prompt sudo pip install scikit-learn

# import sklearn
# import matplotlib
# print(sklearn.__version__)
# from numpy import where
# from sklearn.datasets import make_classification
# from matplotlib import pyplot
# X, y = make_classification(n_samples=10, n_features=7, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# #print(X)
# # create scatter plot for samples from each class
# for class_value in range(2):
#  # get row indexes for samples with this class
#  row_ix = where(y == class_value)
#  # create scatter of these samples
#  pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()
