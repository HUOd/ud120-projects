#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import random

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print("Number of data points:", len(enron_data))
print("Number of features:", len(random.choice(list(enron_data.values()))))

poi_count = 0
for _, value in enron_data.items():
    if value["poi"] == 1:
        poi_count += 1

print("Number of POI:", poi_count)

# poi_names_file = open("../final_project/poi_names.txt", "rb")
# poi_names_file_lines = [line.decode("utf-8").rstrip() for line in poi_names_file.readlines()]
#
# poi_names_count = 0
# for line in poi_names_file_lines:
#     if line.startswith('(n)') or line.startswith('(y)'):
#         name = line[4:].replace(",", "").upper()
#         if name in enron_data and enron_data[name]["poi"] == 1:
#             poi_names_count += 1
#
# print("Number of name of POI:", poi_names_count)

print("Total value of the stock for James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"])
print("Messages from Wesley Colwell to POIs:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("The value of stock options exercised by Jeff Skilling:", enron_data["SKILLING JEFF"]["exercised_stock_options"])
