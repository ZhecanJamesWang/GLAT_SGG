import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# infile = open("train_triplet_freq",'rb')
# new_dict = pickle.load(infile)
# x = []
# y = []
#
# length = len(new_dict.keys())
# counter = 0
# for key in new_dict.keys():
#     # print(new_dict[key])
#     # raise("debug")
#     y.append(new_dict[key])
#     counter += 1
#     if counter % 100 == 0:
#         print('{}/{}'.format(counter, length))
# infile.close()
#
# with open("train_triplet_freq_y", 'wb') as f:
#     pickle.dump(y, f)
# ==========================================================
infile = open("train_triplet_freq_y",'rb')
y = pickle.load(infile)
infile.close()

print("np.max(y): ", np.max(y))
max = np.max(y)
split = 4
unit = int(float(max)/split)

data = y


# bins = []
# for spl in range(split):
#     bins.append(spl * unit)
# bins.append(max)
# print("bins: ", bins)


# digitized = np.digitize(data, bins)
#
# print("digitized: ", digitized)
# print("len(digitized): ", len(digitized))

#print(stats.binned_statistic(data, data, 'sum', bins=2))

# ys = []
# for index in range(len(bins)):
#     y = []
#     bin = bins[index]
#     for d in data:
#         if index == split - 1:
#             if d >= bin and d < max:
#                 y.append(d)
#         else:
#             if d >= bin and d < bins[index + 1]:
#                 y.append(d)
#     ys.append(y)
#
# for y in ys:
#     print(len(y))


# bins:  [0, 451, 902, 1353, max]
bins = [0, 20, 100, max]
total_freq = 0
bin_freqs = []
for i in range(len(bins)-1):
    fig = plt.figure()
    bin_freq = 0
    # i = 0
    x1 = bins[i]
    x2 = bins[i+1]
    print(x1, x2)

    data_return = []
    counter = 0
    length = len(data)
    for i in range(length):
        d = data[i]
        if d >= x1 and d <= x2:
            data_return.append(d)
            bin_freq += d
        counter += 1
        if counter % 5000 == 0:
            print('{}/{}'.format(counter, length))

    total_freq += bin_freq
    bin_freqs.append(bin_freq)
    plt.hist(data_return, bins=100)
    #
    # # h=plt.hist(x,y)
    # # plt.axis([0, 6, 0, 6])
    #
    # # plt.bar(bins, bin_means)
    # #
    # #
    plt.savefig('train_triplet_freq_{}_{}_{}.png'.format(x1, x2, max))
    print('train_triplet_freq_{}_{}_{}.png'.format(x1, x2, max))
    print(len(data_return)/len(data))
    print("bin_freq: ", bin_freq)
print("total_freq: ", total_freq)
print(np.asarray(bin_freqs)/total_freq)