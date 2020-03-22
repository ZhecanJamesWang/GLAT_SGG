import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')

# infile = open("train_triplet_freq",'rb')
#
# new_dict = pickle.load(infile)
# x = []
# y = []
#
# length = len(new_dict.keys())
# counter = 0
#
# for key in new_dict.keys():
#     x.append(key)
#     y.append(new_dict[key])
#     counter += 1
#     # if counter % 100 == 0:
#     #     print('{}/{}'.format(counter, length))

# sorted_list = sorted(zip(x, y), key=lambda pair: pair[1])

# infile.close()
# with open("train_triplet_freq_y", 'wb') as f:
#     pickle.dump(y, f)
#
# max_v = np.max(y)
# print("max_v: ", max_v)
#
# # split = 4
# # unit = int(float(max_v)/split)
#
# # bins = []
# # for spl in range(split):
# #     bins.append(spl * unit)
#
# bins = [0]
#
# triplet_bin = {}
# for x, y in sorted_list:
#     for index in range(len(bins)):
#         bin = bins[index]
#         if index == len(bins) - 1:
#             if y >= bin and y <= max_v:
#                 if index in triplet_bin:
#                     triplet_bin[index].append([x, y])
#                 else:
#                     triplet_bin[index] = [[x, y]]
#         else:
#             if y >= bin and y < bins[index + 1]:
#                 if index in triplet_bin:
#                     triplet_bin[index].append([x, y])
#                 else:
#                     triplet_bin[index] = [[x, y]]
#
# with open("triplet_bin_dict", 'wb') as f:
#     pickle.dump(triplet_bin, f)
# # ==========================================================
# infile = open("triplet_bin_dict",'rb')
# triplet_bin = pickle.load(infile)
# infile.close()

# file = "cache/motif_predcls_2020_0228_1720_dict_pred_list_total"
# file = "cache/motif_predcls_2020_0228_1735_dict_pred_list_total"
# file = "caches/kern_sgcls_2020_0229_0338_dict_pred_list_total.pkl"
# file = "caches/kern_sgcls_2020_0229_0428_dict_pred_list_total.pkl"

# kern
# file = "caches/kern_sgcls_2020_0229_2121.pkl_dict_pred_list_total"
# file = "caches/kern_sgcls_2020_0229_2132.pkl_dict_pred_list_total"

# motif
# file = "cache/motif_predcls_2020_0229_2140_dict_pred_list_total"
# file = "cache/motif_predcls_2020_0229_2204_dict_pred_list_total"

# file = "cache/motif_predcls_2020_0302_1827_dict_pred_list_total"
# file = "cache/motif_predcls_2020_0302_1836_dict_pred_list_total"

# file = "cache/motif_predcls_2020_0303_0404_dict_pred_list_total"
file = "cache/motif_predcls_2020_0303_0403_dict_pred_list_total"

infile = open(file,'rb')

dict_pred_list_total = pickle.load(infile)

# for bin, triplets in triplet_bin.items():
#     bin_recall_num = 0
#     bin_recall_den = 0
#     for triplet in triplets:
#         try:
#             res = dict_pred_list_total[tuple(triplet[0])]
#             bin_recall_num += res[1]
#             bin_recall_den += res[0]
#         except Exception as e:
#             # print(e)
#             pass
#
#     print(float(bin_recall_num)/bin_recall_den)

# dict_pred_list_total

recall_num = 0
recall_den = 0
for triplet, value in dict_pred_list_total.items():
    recall_den += value[0]
    recall_num += value[1]

print(float(recall_num)/recall_den)
