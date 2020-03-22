def max_sum_sub_seq(arr, n):
    max_val = 0
    max_sum_sub_seq_holder = [0] * n

    for i in range(n):
        max_sum_sub_seq_holder[i] = arr[i]

    for i in range(1, n):
        for j in range(i):
            if (arr[i] > arr[j] and max_sum_sub_seq_holder[i] < max_sum_sub_seq_holder[j] + arr[i]):
                max_sum_sub_seq_holder[i] = max_sum_sub_seq_holder[j] + arr[i]

    for i in range(n):
        if max_val < max_sum_sub_seq_holder[i]:
            max_val = max_sum_sub_seq_holder[i]

    return max_val


arrs = [[1, 101, 2, 3, 100, 4, 5], [173, 48, 118, 193, 68, 196], [38, 141, 73, 138, 134, 80, 193], [169, 16],
        [100, 190, 119, 145, 74], [176, 197, 26, 68, 152, 104, 93, 186, 143], [127, 93, 127, 183, 57, 151, 126, 66],
        [139, 107, 110, 80], [196, 85, 106, 134, 137], [123, 175, 184, 198]]

for arr in arrs:
    n = len(arr)
    print(arr)
    # print("max sub increasing sequence ", max_sub_seq(arr))
    print("max sum of sub increasing sequence ", max_sum_sub_seq(arr, n))






