import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# infile = open("train_triplet_freq",'rb')
# infile = open("train_triplet_freq_20200302", 'rb')
infile = open("train_triplet_freq_0303_1724", 'rb')


train_triplet_freq = pickle.load(infile)
x = []
y = []

length = len(train_triplet_freq.keys())
counter = 0

for key in train_triplet_freq.keys():
    x.append(key)
    y.append(train_triplet_freq[key])
    counter += 1
    # if counter % 100 == 0:
    #     print('{}/{}'.format(counter, length))

sorted_list = sorted(zip(x, y), key=lambda pair: pair[1])

# infile.close()
# with open("train_triplet_freq_y", 'wb') as f:
#     pickle.dump(y, f)

max_v = np.max(y)
split = 4
unit = int(float(len(sorted_list))/split)

print("ranking mean recall triplet ")

print("max_v: ", max_v)

bins = []
for spl in range(split):
    bins.append(spl * unit)
# bins.append(max_v)
bins.append(len(sorted_list))

# bins = [0, max_v]
# bins = [1, 10, 100, max_v]
bins = [1, 3, 9, 27, 81, 243, max_v]
# bins = [0, 100, 1000, 10000, 104602]
# bins = [0, 300, 900, 2700, 8100, 24300, 72900, 104602]
print(bins)
triplet_bin = {}
for ranking in range(len(sorted_list)):
    x, y = sorted_list[ranking]
    for index in range(len(bins)):
        bin = bins[index]
        if index == len(bins) - 1:
            break
            # if y >= bin and y <= max_v:
            #     if index in triplet_bin:
            #         triplet_bin[index].append([x, y])
            #     else:
            #         triplet_bin[index] = [[x, y]]
        else:
            if ranking >= bin and ranking <= bins[index + 1]:
                if index in triplet_bin:
                    triplet_bin[index].append([x, y])
                else:
                    triplet_bin[index] = [[x, y]]


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
# files = ["caches/kern_sgcls_2020_0303_1151.pkl_dict_pred_list_total", "caches/kern_sgcls_2020_0303_1153.pkl_dict_pred_list_total"]

# motif
# file = "cache/motif_predcls_2020_0229_2140_dict_pred_list_total"
# file = "cache/motif_predcls_2020_0229_2204_dict_pred_list_total"

# files = ["cache/motif_predcls_2020_0303_0404_dict_pred_list_total", "cache/motif_predcls_2020_0303_0403_dict_pred_list_total"]
#
# stanford
files = ["cache/eval_stanford_glat_sgcls_2020_0303_1220_dict_pred_list_total", "cache/eval_stanford_glat_sgcls_2020_0303_1227_dict_pred_list_total"]

# debiasd kern
# files = ["caches/kern_sgcls_2020_0303_1540.pkl_dict_pred_list_total", "caches/kern_sgcls_2020_0303_1543.pkl_dict_pred_list_total", "caches/kern_sgcls_2020_0303_1153.pkl_dict_pred_list_total"]


# counter = 0
for file in files:
    infile = open(file,'rb')

    dict_pred_list_total = pickle.load(infile)

    pred_predicate_dict = {}

    for pair, value in dict_pred_list_total.items():
        key = pair[-1]
        if key in pred_predicate_dict:
            res = dict_pred_list_total[pair]
            pred_predicate_dict[key][1] += res[1]
            pred_predicate_dict[key][0] += res[0]
        else:
            pred_predicate_dict[key] = dict_pred_list_total[pair]

    stats = ""
    # counter = 0
    for bin, triplets in triplet_bin.items():
        # bin_recall_num = 0
        bin_recall_den = 0
        bin_recall = []
        bin_con_triplet_freq = 0
        for triplet in triplets:
            key = triplet[0][-1]
            if key in pred_predicate_dict:
                res = pred_predicate_dict[key]
                bin_recall.append(float(res[1]) / res[0])
                # bin_recall_den += res[0]
            # bin_con_triplet_freq += train_predicate_freq[key]
            # counter += 1

        # print("\n")
        print("bin {}".format(bin))
        print(np.mean(bin_recall))

        stats += ("bin {}".format(bin) + "\n")
        stats += ("# of triplet: {}, triplet_occur train: {} \n".format(len(triplets), bin_con_triplet_freq))
        print("counter: ", counter)
        stats += ("triplet occur test: {} \n".format(bin_recall_den))

    # counter += 1
    print("\n")

print(stats)
print("\n")
