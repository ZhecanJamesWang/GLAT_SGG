def max_palindrome(s):

    if len(s) == 1:
        return s

    max_string = ""
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            if s[i:j] == s[i:j][::-1] and len(s[i:j])>len(max_string):
                max_string = s[i:j]

    return len(max_string)


arrs = ["babad", "fox", "amaury", "hatee", "wibisono", "neveroddoreven", "derekchen", "programming",
        "akusukarajawalibilawajarakusuka", "wapapapapapapow"]

for arr in arrs:
    print(arr)
    print("max sum of sub increasing sequence ", max_palindrome(arr))
