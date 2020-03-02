import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

infile = open("train_triplet_freq",'rb')

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
split = 10
unit = int(float(max_v)/split)

print("max_v: ", max_v)

bins = []
for spl in range(split):
    bins.append(spl * unit)

bins = [1, 10, 100, 1000]
# bins = [1, 3, 9, 27, 81, 243, 729]

triplet_bin = {}
for x, y in sorted_list:
    for index in range(len(bins)):
        bin = bins[index]
        if index == len(bins) - 1:
            if y >= bin and y <= max_v:
                if index in triplet_bin:
                    triplet_bin[index].append([x, y])
                else:
                    triplet_bin[index] = [[x, y]]
        else:
            if y >= bin and y < bins[index + 1]:
                if index in triplet_bin:
                    triplet_bin[index].append([x, y])
                else:
                    triplet_bin[index] = [[x, y]]

with open("triplet_bin_dict", 'wb') as f:
    pickle.dump(triplet_bin, f)
# ==========================================================
infile = open("triplet_bin_dict",'rb')
triplet_bin = pickle.load(infile)
infile.close()

# file = "cache/motif_predcls_2020_0228_1720_dict_pred_list_total"
# file = "cache/motif_predcls_2020_0228_1735_dict_pred_list_total"
# file = "caches/kern_sgcls_2020_0229_0338_dict_pred_list_total.pkl"
# file = "caches/kern_sgcls_2020_0229_0428_dict_pred_list_total.pkl"

# kern
# file = "caches/kern_sgcls_2020_0229_2121.pkl_dict_pred_list_total"
# file = "caches/kern_sgcls_2020_0229_2132.pkl_dict_pred_list_total"

# motif
# file = "cache/motif_predcls_2020_0229_2140_dict_pred_list_total"
file = "cache/motif_predcls_2020_0229_2204_dict_pred_list_total"



infile = open(file,'rb')

dict_pred_list_total = pickle.load(infile)

stats =""
for bin, triplets in triplet_bin.items():
    bin_recall_num = 0
    bin_recall_den = 0
    bin_con_triplet_freq = 0

    for triplet in triplets:
        try:
            key = tuple(triplet[0])
            res = dict_pred_list_total[key]
            bin_recall_num += res[1]
            bin_recall_den += res[0]
            bin_con_triplet_freq += train_triplet_freq[key]
        except Exception as e:
            # print(e)
            pass
    print("bin {}".format(bin))
    print(float(bin_recall_num)/bin_recall_den)
    stats += ("bin {}".format(bin) + "\n")
    stats += ("# of triplet: {}, triplet_occur: {}".format(len(triplets), bin_con_triplet_freq) + "\n")

print(stats)
