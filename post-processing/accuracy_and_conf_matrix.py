# python script to get test accuracy and confusion matrix

import numpy as np
import json

import time

num_classes = 6

# import json
file = open('labels.txt', 'r')
true_labels_list = []
myline = file.readline()
mylist = myline.split()
true_labels_list.append(mylist)
while myline:
    myline = file.readline()
    mylist = myline.split()
    true_labels_list.append(mylist)
# data = json.load(file)
file.close()

predicted_labels_as_ints = np.array([0]*(len(true_labels_list) -1))
true_labels_as_ints = np.array([0]*(len(true_labels_list) -1))

file2 = open('predictions.json', 'r')
predicted_data = json.load(file2)
file2.close()

for i in range(len(true_labels_list) - 1):

    true_labels_as_ints[i] = int(true_labels_list[i][0])
    predicted_labels_as_ints[i] = predicted_data[str(i)]['class'][0]# TODO

num_correct = 0
for i in range(len(predicted_labels_as_ints)):
    if predicted_labels_as_ints[i] == true_labels_as_ints[i]:
        num_correct += 1

print(f"original accuracy = {num_correct/len(predicted_labels_as_ints)}")

confusion_matrix = np.array([[0]*num_classes]*num_classes)

for i in range(len(predicted_labels_as_ints)):
    # column = actual class
    # row = predicted class
    column = true_labels_as_ints[i]
    row = predicted_labels_as_ints[i]
    confusion_matrix[row][column] += 1

# get the accuracy without the 10^15 quench
# n_classes = 6, but we skip class 0
num_correct_exclude = 0.0
num_total_exclude = 0.0
for i in range(1, 6, 1):
    num_correct_exclude += confusion_matrix[i][i]
    for j in range(1, 6, 1):
        # print(f"i = {i}, j = {j}")
        # print(confusion_matrix[i][j])
        num_total_exclude += confusion_matrix[i][j]

# print(f"num_total_exclude = {num_total_exclude}")
# print(f"num_correct_exclude = {num_correct_exclude}")
print(f"accuracy without 10^15 quench = {num_correct_exclude/num_total_exclude}")

# normalize confusion matrix without using explicit for loop
tic = time.time()
column_sums2 = confusion_matrix.sum(axis=0)
# print(column_sums2)
norm_confusion_matrix = confusion_matrix/column_sums2
print("Normalized confusion matrix")
print(norm_confusion_matrix)
toc = time.time()
# print(toc-tic)