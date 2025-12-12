from collections import Counter, defaultdict, deque
import heapq
import  re

class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None
def reverse_string(string):
    return string[::-1]
print(reverse_string("hello"))

def two_sum(nums, target):
    seen={};
    for i,x in enumerate(nums):
        need = target - x
        if need in seen: return [seen[need],i]
    return None
print(two_sum([1,2,3,4,5,6,7,8,9], 9))

def binary_search(nums, target):
    low, high = 0, len(nums)-1
    while low <= high:
        mid = (low+high)//2
        if nums[mid] == target: return mid
        elif nums[mid] < target:
            low = mid + 1
        elif nums[mid] > target:
            high = mid - 1
    return -1
print(binary_search([1,2,3,4,5,6,7,8,9], 9))


def max_subarray(nums):
    best=current=nums[0]
    for x in nums[1:]:
        current=max(x,current+x)
        best=max(best,current)
    return best


def merger_intervals(intervals):
    if not intervals: return []
    intervals.sort()
    result = [intervals[0][:]]
    for interval in intervals[1:]:
        last = result[-1]
        if interval[0]<=last[1]:
            last[1] = max(interval[1],last[1])
        else:
            result.append(interval)

    return result

print(merger_intervals([1,2,3,4,5,6,7,8,9]))

def longest_common_substring(strs):
    seen = {}
    left=best=0
    for right, ch in enumerate(strs):
        if ch in seen and seen[ch]>=left:
            left=seen[ch]+1
        seen[ch]=right
        best = max(best, right-left+1)
    return best


import heapq

def kth_largets(nums, k):
    return heapq.nlargest(k,nums)[-1]


from collections import deque
def bfs(graph, start):
    queue = deque([start]); seen = {start}; order=[]
    while queue:
        u = queue.popleft(); order.append(u)
        for v in graph[u]:
            if v not in seen:
                seen.add(v); queue.append(v)
    return order


from typing import Optional,List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

