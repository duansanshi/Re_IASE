x = int(input())

# for i in range(x):
#     (len,q) = map(int,input().split())
#     str1 = input()
#     str2 = input()
#     for i in range(q):
#         (l,r) = map(int,input().split())
#         list1=[0]*26
#         list2=[0]*26
#         times = 0
#         for j in range(l-1,r):
#             list1[ord(str1[j])-ord('a')] += 1
#             list2[ord(str2[j])-ord('a')] += 1
#         for j in range(26):
#             times += abs(list1[j]-list2[j])
#         times//=2
#         print(times)

for i in range(x):
    (len,q) = map(int,input().split())
    str1 = input()
    str2 = input()
    for j in range(q):     
        (l,r) = map(int,input().split())
        str1_q = sorted(str1[l-1:r])
        str2_q = sorted(str2[l-1:r])
        k = 0
        m = 0
        cnt = 0
        while k<r-l+1:
            if str1_q[k]==str2_q[m]:
                k+=1
                m+=1
                continue
            else:
                k+=1
                cnt+=1
        print(cnt)
            