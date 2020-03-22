
def max_coins(nums):

    nums = [1] + nums + [1]
    n = len(nums)

    memo = [[0] * n for _ in range(n)]

    for left in range(n - 2, -1, -1):
        for right in range(left + 2, n):
            sub = []
            # print(left, right)
            for i in range(left + 1, right):
                # print(i)
                score = nums[left] * nums[i] * nums[right] + memo[left][i] + memo[i][right]
                sub.append(score)
            memo[left][right] = max(sub)

    return memo[0][n - 1]


arrs = [[3, 1, 5, 8], [1, 5], [46, 22, 60, 12, 27], [2, 8, 2], [1, 2, 3, 3],
        [], [7, 2, 7, 4, 9, 6], [7,9,8,0,7,1,3,5,5,2],
        [21, 12, 0, 47, 33, 49, 1, 14, 17, 16, 18, 37, 7, 1, 25, 46, 24, 9],
        [15, 40, 9, 0, 45, 20, 0, 67, 0, 0, 82, 0]]
ans = [167, 10, 155968, 40, 30, 0, 1218, 1582, 479123, 521842]

for index in range(len(arrs)):
    arr = arrs[index]
    an = ans[index]
    print(arr)
    print("expected: ", an)
    print("max coins ", max_coins(arr))


# i. Input: ⟨3, 1, 5, 8⟩ Output: 167
# ii. Input: ⟨1, 5⟩ Output: 10
# iii. Input: ⟨46, 22, 60, 12, 27⟩ Output: 155968
# iv. Input: ⟨2, 8, 2⟩ Output: 40
# v. Input: ⟨1, 2, 3, 3⟩ Output: 30
# vi. Input: ⟨⟩ (n = 0) Output: 0
# vii. Input: ⟨7, 2, 7, 4, 9, 6⟩ Output: 1218
# viii. Input: ⟨7,9,8,0,7,1,3,5,5,2⟩ Output: 1582
# ix. Input: ⟨21, 12, 0, 47, 33, 49, 1, 14, 17, 16, 18, 37, 7, 1, 25, 46, 24, 9⟩ Output: 479123
# x. Input: ⟨15, 40, 9, 0, 45, 20, 0, 67, 0, 0, 82, 0⟩ Output: 521842