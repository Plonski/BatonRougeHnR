__author__ = 'wildcat'

import csv
from sklearn.linear_model import LogisticRegression
import numpy as np

count = 0
total_array = []
test_hnr_booleans = []
test_total_array = []

class Tokenize():

    def __init__(self):
        pass

    def check_subset(self, array):
        if -1 in array:
            return -1

    def road_type(self, cell):
        #
        # column 12
        #
        if cell == "CITY STREET":
            return 1
        if cell == "OFF ROAD/ PRIVATE PROPERTY":
            return 2
        if cell == "INTERSTATE":
            return 3
        if cell == "U.S. HWY":
            return 4
        if cell == "STATE HWY":
            return 5
        if cell == "PARISH ROAD":
            return 6
        else:
            return 7

    def time(self, cell):
        #
        # column 2
        #
        if cell[6] == "P":
            time = int(cell[0:2]) + 12 + 1
        else:
            time = int(cell[0:2]) + 1
        return time

    def address_to_coords(self):
        pass

    def crash_month(self, cell):
        #
        # column 1
        #
        if cell[0:2] == "01":
            return 1
        if cell[0:2] == "02":
            return 2
        if cell[0:2] == "03":
            return 3
        if cell[0:2] == "04":
            return 4
        if cell[0:2] == "05":
            return 5
        if cell[0:2] == "06":
            return 6
        if cell[0:2] == "07":
            return 7
        if cell[0:2] == "08":
            return 8
        if cell[0:2] == "09":
            return 9
        if cell[0:2] == "10":
            return 10
        if cell[0:2] == "11":
            return 11
        if cell[0:2] == "12":
            return 12

    def cars_involved(self, cell):
        #
        # column 3
        #
        return int(cell)

    def district(self, cell):
        if len(cell) == 1 and cell.isdigit():
            return cell
        else:
            return -1

    def zone(self, cell):
        if cell == "A":
            return "1"
        if cell == "B":
            return "2"
        if cell == "C":
            return "3"
        if cell == "D":
            return "4"
        if cell == "E":
            return "5"
        if cell == "F":
            return "6"
        if cell == "G":
            return "7"
        else:
            return -1

    def subzone(self, cell):
        if len(cell) == 1 and cell.isdigit():
            return cell
        else:
            return -1

    def fatality_involved(self, cell):
        #
        # column 15
        #
        if len(cell) == 1:
            return 1
        else:
            return 0

    def injury_involved(self, cell):
        #
        # column 16
        #
        if len(cell) == 1:
            return 1
        else:
            return 0

    def pedestrian_involved(self, cell):
        #
        # column 17
        #
        if len(cell) == 1:
            return 1
        else:
            return 0

    def at_intersection(self, cell):
        #
        # column 18
        #
        if len(cell) == 1:
            return 1
        else:
            return 0

    def manner_of_collision(self, cell):
        #
        # column 20
        #
        if cell == "RIGHT ANGLE":
            return 1
        if cell == "REAR END":
            return 2
        if cell == "LEFT TURN":
            return 3
        if cell == "SIDESWIPE SAME":
            return 4
        if cell == "HEAD-ON":
            return 5
        if cell == "SIDESWIPE OPPOSITE":
            return 6
        if cell == "OTHER":
            return 7
        if cell == "RIGHT TURN":
            return 8
        if cell == "NON-COLLISION WITH MOTOR VEHICLE":
            return 9
        else:
            return -1

    def surface_condition(self, cell):
        #
        # column 21
        #
        if cell == "DRY":
            return 1
        if cell == "WET":
            return 2
        if cell == "UNKNOWN":
            return 3
        if cell == "OTHER":
            return 4
        if cell == "ICE":
            return 5
        if cell == "SNOW/SLUSH":
            return 6
        else:
            return -1

    def surface_type(self, cell):
        #
        # column 22
        #
        if cell == "BLACK TOP":
            return 1
        if cell == "CONCRETE":
            return 2
        if cell == "GRAVEL":
            return 3
        if cell == "OTHER":
            return 4
        if cell == "BRICK":
            return 5
        if cell == "MUD":
            return 6
        if cell == "DIRT":
            return 7
        else:
            return -1

    def road_conditions(self, cell):
        #
        # column 23
        #
        if cell == "NO ABNORMALITIES":
            return 1
        if cell == "SHOULDER ABNORMALITIES":
            return 2
        if cell == "WATER ON ROADWAY":
            return 3
        if cell == "ROAD CONSTRUCTION" or "CONSTRUCTION - NO WARNING":
            return 4
        if cell == "OTHER":
            return 5
        if cell == "ANIMAL IN ROADWAY":
            return 6
        if cell == "PREVIOUS CRASH":
            return 7
        if cell == "OBJECT IN ROADWAY":
            return 8
        if cell == "HOLES":
            return 9
        if cell == "BUMPS":
            return 10
        if cell == "DEEP RUTS":
            return 11
        if cell == "LOOSE SURFACE MATERIAL":
            return 12
        if cell == "OVERHEAD CLEARANCE LIMITED":
            return 13
        else:
            return -1

    def road_direction_type(self, cell):
        #
        # column 24
        #
        if cell == "TWO-WAY ROAD WITH NO PHYSICAL SEPARATION":
            return 1
        if cell == "TOW-WAY ROAD WITH A PHYSICAL SEPARATION" or "TWO-WAY ROAD WITH A PHYSICAL BARRIER":
            return 2
        if cell == "ONE-WAY ROAD":
            return 3
        if cell == "OTHER":
            return 4
        else:
            return -1

    def road_alignment(self, cell):
        #
        # column 25
        #
        if cell == "STRAIGHT-LEVEL":
            return 1
        if cell == "OTHER":
            return 2
        if cell == "STRAIGHT-LEVEL ELEVATED":
            return 3
        if cell == "CURVE-LEVEL ELEVATED":
            return 4
        if cell == "CURVE- LEFT":
            return 5
        if cell == "ON GRADE-CURVE":
            return 6
        if cell == "HILLCREST-CURVE":
            return 7
        else:
            return -1

    def crash_reason(self, cell):
        #
        # column 26
        #
        if cell == "VIOLATIONS":
            return 1
        if cell == "CONDITION OF DRIVER":
            return 2
        if cell == "MOVEMENT PRIOR TO CRASH":
            return 3
        if cell == "VEHICLE CONDITIONS":
            return 4
        if cell == "PEDESTRIAN ACTIONS" or "CONDITION OF PEDESTRIAN":
            return 5
        if cell == "OTHER":
            return 6
        if cell == "KIND OF LOCATION":
            return 7
        if cell == "ROADWAY CONDITION" or "ROAD_SURFACE":
            return 8
        if cell == "VISION OBSCUREMENTS":
            return 9
        if cell == "WEATHER":
            return 10
        if cell == "TRAFFIC CONTROL":
            return 11
        if cell == "OTHER":
            return 12
        else:
            return -1

    def secondary_reason(self, cell):
        #
        # column 27
        #
        if len(cell) == 0:
            return 1
        if cell == "VIOLATIONS":
            return 2
        if cell == "CONDITION OF DRIVER":
            return 3
        if cell == "MOVEMENT PRIOR TO CRASH":
            return 4
        if cell == "VEHICLE CONDITIONS":
            return 5
        if cell == "PEDESTRIAN ACTIONS" or "CONDITION OF PEDESTRIAN":
            return 6
        if cell == "OTHER":
            return 7
        if cell == "KIND OF LOCATION":
            return 8
        if cell == "ROADWAY CONDITION" or "ROAD_SURFACE":
            return 9
        if cell == "VISION OBSCUREMENTS":
            return 10
        if cell == "WEATHER":
            return 11
        if cell == "TRAFFIC CONTROL":
            return 11
        if cell == "OTHER":
            return 12
        else:
            return -1

    def weather(self, cell):
        #
        # column 28
        #
        if cell == "CLOUDY":
            return 1
        if cell == "CLEAR":
            return 2
        if cell == "RAIN":
            return 3
        if cell == "FOG/SMOKE":
            return 4
        if cell == "SLEET/HAIL":
            return 5
        if cell == "SEVERE CROSSWIND":
            return 6
        if cell == "SNOW":
            return 7
        if cell == "UNKNOWN" or "":
            return 8
        else:
            return -1

    def location_kind(self, cell):
        #
        # column 29
        #
        if cell == "RESIDENTIAL DISTRICT":
            return 1
        if cell == 'RESIDENTIAL SCATTERED':
            return 2
        if cell == "BUSINESS CONTINUOUS":
            return 3
        if cell == "OTHER":
            return 4
        if cell == "MANUFACTURING OF INDUSTRAIL":
            return 5
        if cell == "OPEN COUNTRY":
            return 6
        if cell == "BUSINESS, MIXED RESIDENTIAL":
            return 7
        if cell == "SCHOOL OR PLAYGROUND":
            return 8
        else:
            return -1

    def road_relation(self, cell):
        if cell == "ON ROADWAY":
            return 1
        if cell == "OTHER":
            return 2
        if cell == "SHOULDER":
            return 3
        if cell == "BEYOND SHOULDER - RIGHT":
            return 4
        if cell == "BEYOND SHOULDER - LEFT":
            return 5
        if cell == "MEDIAN":
            return 6
        if cell == "UNKNOWN":
            return 7
        if cell == "OPEN COUNTRY":
            return 8
        else:
            return -1


    def road_access(self, cell):
        #
        # column 31
        #
        if cell == "NO CONTROL (UNLIMITED ACCESS TO ROADWAY)":
            return 1
        if cell == "FULL CONTROL (ONLY RAMP ENTRANCE and EXIT)":
            return 2
        if cell == "PARTIAL CONTROL LIMITED ACCESS TO ROADWAY":
            return 3
        if cell == "OTHER":
            return 4
        else:
            return -1

    def lighting(self, cell):
        #
        # column 32
        #
        if cell == "DAYLIGHT":
            return 1
        if cell == "DARK - CONTINUOUS STREET":
            return 2
        if cell == "UNKNOWN":
            return 3
        if cell == "DARK - STREET LIGHT AT INTERSECTION ONLY":
            return 4
        if cell == "OTHER":
            return 5
        if cell == "DUSK":
            return 6
        if cell == "DAWN":
            return 7
        if cell == "DARK - NO STREET":
            return 8
        else:
            return -1

    def hit_and_run(self, cell):
        #
        # column 13
        #
        if len(cell) == 1:
            return 1
        else:
            return 0




b = 0

token = Tokenize()
ct = 0
y = []
hnr_booleans = []

hits = []

with open("Baton_Rouge_Traffic_Incidents.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    next(spamreader, None)
    for row in spamreader:
        if row[0] != '"' and row[0] != "BATON ROUGE" and len(row) != 2:
            ct = ct + 1
        else:
            continue

        instance_array = []
        time = 0

        if row[30] not in hits:
            hits.append(row[30])

        # Gets the month the crash occurred in
        instance_array.append(int(row[1][0:2]))
        # Gets the time the crash occurred
        instance_array.append(token.time(row[2]))
        # Gets the number of vehicles that were involved
        instance_array.append(int(row[3]))

        # Gets the district/zone/subzone the crash occurred in
        if(token.district(row[4]) != -1 and token.zone(row[5]) != -1 and token.subzone(row[6]) != -1):
            instance_array.append(int(token.district(row[4]) + token.zone(row[5]) + token.subzone(row[6])))
        else:
            continue

        # Gets the road type the crash occurred on
        instance_array.append(token.road_type(row[12]))
        # Gets if a fatality occurred
        instance_array.append(token.fatality_involved(row[15]))
        # Gets if a injury occurred
        instance_array.append(token.injury_involved(row[16]))
        # Gets if a pedestrian was a involved
        instance_array.append(token.pedestrian_involved(row[17]))
        # Gets if the crash occurred at a intersection
        instance_array.append(token.at_intersection(row[18]))
        # Gets the manner of the crash
        instance_array.append(token.manner_of_collision(row[20]))
        # Gets the surface condition
        instance_array.append(token.surface_condition(row[21]))
        # Gets the surface type
        instance_array.append(token.surface_type(row[22]))
        # Gets the road condition
        instance_array.append(token.road_conditions(row[23]))
        # Gets the road type
        instance_array.append(token.road_direction_type(row[24]))
        # Gets the road alignment
        instance_array.append(token.road_alignment(row[25]))
        # Gets the primary reason
        instance_array.append(token.crash_reason(row[26]))
        # Gets the secondary reason
        instance_array.append(token.secondary_reason(row[27]))
        # Gets the weather
        instance_array.append(token.weather(row[28]))
        # Gets the location type
        instance_array.append(token.location_kind(row[29]))
        # Gets the access to road
        instance_array.append(token.road_access(row[31]))
        # Gets the lighting
        instance_array.append(token.lighting(row[32]))

        # Gets if a HNR occurred

        hnr_booleans.append(token.hit_and_run(row[13]))
        total_array.append(instance_array)
        count = count + 1



x = LogisticRegression()


X = np.array(total_array)
Y_train = np.array(hnr_booleans)
x.fit(X,Y_train)

c = 0

#
# Month, hour, # of vehicles involved, district/zone/subzone, road type, fatality, injury, pedestrian, intersection,
# manner of collision, surface condition, surface type, road condition, road direction type, road alignment,
# crash reason, secondary reason, weather, location kind, road access, lighting
#

print("The demo example values used in class was:\n"
      "3, 19, 2, 232, 2, 0, 0, 0, 0, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1")
print("Which means: [March, 7 PM, 2 cars involved, District 2 zone c subzone 2, off-road/private property, no fatality"
      ", no injury, no pedestrian, no intersection, rearend, wet road, concrete, no road problems, Two way road with a"
      " physical separation, level road , violations, no secondary reason, clear, residential district, Unlimited access to road, daylight]")

while c < 100:
    print("\nType exit to leave the demo and output the evaluation of the logistic regression")
    vals = input("Enter a instance attribute set").split(",")
    #print(vals)
    integer_vals = []

    if vals[0] == "exit" or vals[0] == "Exit":
        print("Demo is exiting. Total data results outputting. Please wait. (This may take 1-2 minutes)")
        break
    try:
        for num in vals:
            integer_vals.append(int(num))
    except:
        print("Input error, enter the 21 attributes separated by commas")
        continue
    try:
        print("Probability a HNR does not occur vs the probability a HNR does occur")
        print(x.predict_proba([integer_vals]))
    except:
        print("Needs 21 attributes. Input error")


















std_list = []

probability_list = []
double_probability_list = []
binary_prediction = []

positive_array = []

for b in total_array:
    #print("test")
    binary_prediction.append(x.predict([b]))
    probability_list.append(x.predict_proba([b])[0][0])
    std_list.append(x.predict_proba([b])[0][1])
    if(x.predict_proba([b])[0][1] > .50):
        positive_array.append(b)

new_bools = []




# Flips the booleans in order to accommodate brier score loss
for h in hnr_booleans:
    if h == 0:
        new_bools.append(1)
    else:
        new_bools.append(0)

from sklearn.metrics import brier_score_loss, average_precision_score, accuracy_score

print("\nAccuracy score as defined by\n "
      "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score")
print(accuracy_score(hnr_booleans, binary_prediction))
print("\nBrier score loss as defined by\n "
      "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss")
print(brier_score_loss(new_bools, probability_list))
print("\nArea under the PR-Curver\n "
      "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score")
print(average_precision_score(new_bools, probability_list, average='micro'))


import numpy
print("\nMean across the percentages")
print(numpy.mean(std_list))
print("\nSTD of the likelihood")
print(numpy.std(std_list))
print("\nVariance")
print(numpy.var(std_list))

