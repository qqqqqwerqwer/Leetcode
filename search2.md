# 对撞指针

## 示例1 two sum
问题描述：给出一个整型数组nums，返回这个数组中两个数字的索引值i和j，使得nums[i] + nums[j]等于一个给定的target值，两个索引不能相等。

如：nums= [2,7,11,15],target=9 返回[0,1]
     
思路分析：
因为问题本身不是有序的，因此需要对原来的数组进行一次排序，排序后就可以用O(n)的指针对撞进行解决。

在排序前先使用一个额外的数组拷贝一份原来的数组，对于两个相同元素的索引问题，使用一个bool型变量辅助将两个索引都找到。

```python
### two sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums = list(enumerate(nums)
        nums.sort(key = lambda x:x[1]) # 对数组进行排序
        # 对撞指针
        l,r = 0, len(nums)-1
        while l < r:
            if nums[l][1] + nums[r][1] == target:
                return nums[l][0],nums[r][0]
            elif nums[l][1] + nums[r][1] < target:
                l += 1
            else:
                r -= 1
```

## 示例2 three sum
问题描述：给出一个整型数组，寻找其中的所有不同的三元组(a,b,c)，使得a+b+c=0

注意：答案中不可以包含重复的三元组。

如：nums = [-1, 0, 1, 2, -1, -4]，

结果为：[[-1, 0, 1],[-1, -1, 2]]
     
思路分析：
开始时对nums数组进行排序，排序后，当第一次遍历的指针k遇到下一个和前一个指向的值重复时，就将其跳过。为了方便计算，在第二层循环中，可以使用对撞指针。

在里层循环中，也要考虑重复值的情况，因此当值相等时，再次移动指针时，需要保证其指向的值和前一次指向的值不重复。

调整下遍历的范围，因为设了3个索引：i，l，r。边界情况下，r索引指向len-1, l指向len-2，索引i遍历的边界为len-3，故for循环是从0到len-2。

```python
### three sum
class Solution:
    def threeSum(self, nums: [int]) -> [[int]]:
        nums.sort() # 排序
        res = []
        for i in range(len(nums)-2): # 从0遍历到len-2
            # 特例
            if nums[i] > 0: break
            # 重复值情况
            if i > 0 and nums[i] == nums[i-1]: continue
            l,r = i+1, len(nums)-1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum == 0:
                    res.append([nums[i],nums[l],nums[r]])
                    l += 1
                    r -= 1
                    # 重复值
                    while l < r and nums[l] == nums[l-1]: l += 1
                    while l < r and nums[r] == nums[r+1]: r -= 1
                elif sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
```

## 示例3 four sum
问题描述：给出一个整形数组，寻找其中的所有不同的四元组(a,b,c,d)，使得a+b+c+d等于一个给定的数字target。

如:
nums = [1, 0, -1, 0, -2, 2]，target = 0

结果为：
[[-1,  0, 0, 1],[-2, -1, 1, 2],[-2,  0, 0, 2]]
     
思路分析：
首先排序，接着从[0,len-1]遍历i，跳过i的重复元素，再在[i+1,len-1]中遍历j，得到i，j后，再选择首尾的l和r，通过对撞指针的思路，四数和大的话r--，小的话l++,相等的话纳入结果list，最后返回。

```python
### four sum
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort() # 排序
        res = []
        # 特例
        if len(nums) < 4: return res
        if len(nums) == 4 and sum(nums) == target:
            res.append(nums)
            return res
        # 多套一层循环，注意遍历的边界
        for i in range(len(nums)-3):
            if i > 0 and nums[i] == nums[i-1]: continue
            for j in range(i+1,len(nums)-2):
                if j > i+1 and nums[j] == nums[j-1]: continue
                l,r = j+1, len(nums)-1
                while l < r:
                    sum_value = nums[i] + nums[j] + nums[l] + nums[r]
                    if sum_value == target:
                        res.append([nums[i],nums[j],nums[l],nums[r]])
                        l += 1
                        r -= 1
                        while l < r and nums[l] == nums[l-1]: l += 1
                        while l < r and nums[r] == nums[r+1]: r -= 1
                    elif sum_value < target:
                        l += 1
                    else:
                        r -= 1
        return res
```

## 示例4 three number closest
问题描述：给出一个整形数组，寻找其中的三个元素a,b,c，使得a+b+c的值最接近另外一个给定的数字target。

如：给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
     
思路分析：
开始时可以随机设定一个三个数的和为结果值，在每次比较中，先判断三个数的和是否和target相等，如果相等直接返回和。

如果不相等，则判断三个数的和与target的差是否小于这个结果值时，如果小于则进行则进行替换，并保存和的结果值。

```python
### three number closest
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort() # 排序
        diff = abs(nums[0]+nums[1]+nums[2]-target) # 初始化差值
        res = nums[0] + nums[1] + nums[2]
        for i in range(len(nums)):
            l,r = i+1,len(nums)-1
            t = target - nums[i]
            while l < r:
                if nums[l] + nums[r] == t:
                    return nums[i] + t
                else:
                    if abs(nums[l]+nums[r]-t) < diff:
                        diff = abs(nums[l]+nums[r]-t) # 更新差值
                        res = nums[i]+nums[l]+nums[r]
                    if nums[l]+nums[r] < t:
                        l += 1
                    else:
                        r -= 1
        return res
```

## 示例5 four sum 2.0
问题描述：给出四个整形数组A,B,C,D,寻找有多少i,j,k,l的组合,使得A[i]+B[j]+C[k]+D[l]=0。其中,A,B,C,D中均含有相同的元素个数N，且0<=N<=500；

输入:

A = [ 1, 2] B = [-2,-1] C = [-1, 2] D = [ 0, 2]

输出:2
     
思路分析：
可以考虑把D数组中的元素都放入查找表，然后遍历前三个数组，判断target减去每个元素后的值是否在查找表中存在，存在的话，把结果值加1。

考虑到数组中可能存在重复的元素，而重复的元素属于不同的情况，因此用dict存储，最后的结果值加上dict相应key的value，

```python
### word pattern
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        from collections import Counter
        record = Counter()
        for i in range(len(D)):
            record[D[i]] += 1
        res = 0 
        for i in range(len(A)):
            for j in range(len(B)):
                for k in range(len(C)):
                    num_find = 0-A[i]-B[j]-C[k]
                    if record.get(num_find) != None:
                        res += record(num_find)
        return res
```

## 示例6 字母异位词分组
问题描述：给出一个字符串数组，将其中所有可以通过颠倒字符顺序产生相同结果的单词进行分组。

示例:
输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:[["ate","eat","tea"],["nat","tan"],["bat"]]
     
思路分析：
将字符串统一排序，异位词排序后的字符串，显然都是相同的。那么就可以把其当作key，把遍历的数组中的异位词当作value，对字典进行赋值，进而遍历字典的value，得到结果list。

```python
### 字母异位词分组
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        strs_dict = defaultdict(list)
        for str in strs:
            key = ''.join(sorted(list(str))) # 排序并将list转换为字符串
            strs_dict[key] += str.split(',') # 将字符串整个转换为list中的一项
        return [v for v in strs_dict.values()]
```

## 示例7 回旋镖的数量
问题描述：给出一个平面上的n个点，寻找存在多少个由这些点构成的三元组(i,j,k)，使得i,j两点的距离等于i,k两点的距离。

其中n最多为500,且所有的点坐标的范围在[-10000,10000]之间。

输入:
[[0,0],[1,0],[2,0]]

输出:
2
解释:
两个结果为： [[1,0],[0,0],[2,0]] 和 [[1,0],[2,0],[0,0]]
     
思路分析：
当i,j两点距离等于i,k时，用查找表的思路，等价于：对距离key(i,j或i,k的距离)，其值value(个数)为2。

那么就可以做一个查找表，用来查找相同距离key的个数value是多少。遍历每一个节点i，扫描得到其他点到节点i的距离，在查找表中，对应的键就是距离的值，对应的值就是距离值得个数。

```python
### 回旋镖的数量
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        from collections import Counter
        def f(x1, y1):
            d = Counter((x2 - x1) ** 2 + (y2 - y1) ** 2 for x2, y2 in points)
            return sum(t * (t-1) for t in d.values())
        return sum(f(x1, y1) for x1, y1 in points)
```

## 示例8 Max Points on a Line
问题描述：给定一个二维平面，平面上有 n 个点，求最多有多少个点在同一条直线上。

示例 1:
输入: [[1,1],[2,2],[3,3]]
输出: 3

示例 2:
输入: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出: 4

思路分析：
查找相同斜率key的个数value是多少。

```python
### Max Points on a Line
class Solution:
    def maxPoints(self,points):
        # 特例
        if len(points) <= 1:
            return len(points)
        res = 0
        from collections import defaultdict
        for i in range(len(points)):
            record = defaultdict(int)
            samepoint = 0 # 相同点
            for j in range(len(points)):
                if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                    samepoint += 1
                else:
                    record[self.get_Slope(points,i,j)] += 1
            for v in record.values():
                res = max(res, v+samepoint)
            res = max(res, samepoint)
        return res
    def get_Slope(self,points,i,j):
        if points[i][1] - points[j][1] == 0:
            return float('Inf')
        else:
            return (points[i][0] - points[j][0]) / (points[i][1] - points[j][1])
```
# 滑动数组
## 示例1 存在的重复元素
问题描述：给出一个整形数组nums和一个整数k，是否存在索引i和j，使得nums[i]==nums[j]，且i和J之间的差不超过k。

示例1:
输入: nums = [1,2,3,1], k = 3
输出: true

示例 2:
输入: nums = [1,2,3,1,2,3], k = 2
输出: false

思路分析：
这道题目可以考虑使用滑动数组来解决：固定滑动数组的长度为K+1，当这个滑动数组内如果能找到两个元素的值相等，就可以保证两个元素的索引的差是小于等于k的。

如果当前的滑动数组中没有元素相同，就右移滑动数组的右边界r,同时将左边界l右移。查看r++的元素是否在l右移过后的数组里，如果不在就将其添加数组，在的话返回true表示两元素相等。

```python
### 存在的重复元素
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        record = set() # 用set作为数据结构
        for i in range(len(nums)):
            if nums[i] in record:
                return True
            record.add(nums[i])
            if len(record) == k+1:
                record.remove(nums[i-k])
        return False
```

## 示例2 分割数组的最大值
问题描述：给定一个整数数组，判断数组中是否有两个不同的索引 i 和 j，使得nums [i] 和nums [j]的差的绝对值最大为 t，并且 i 和 j 之间的差的绝对值最大为 ķ。

示例 1:

输入: nums = [1,2,3,1], k = 3, t = 0

输出: true

示例 2:

输入: nums = [1,0,1,1], k = 1, t = 2

输出: true

示例 3:

输入: nums = [1,5,9,1,5,9], k = 2, t = 3

输出: false

思路分析：
将索引的差值固定，于是问题和上道一样，同样转化为了固定长度K+1的滑动窗口内，是否存在两个值的差距不超过t，考虑使用滑动窗口的思想来解决。

对于有序数组可以结合二分查找。

```python
### 分割数组的最大值
class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t) -> bool:
        record = set() # 用set作为数据结构
        for i in range(len(nums)):
            if len(record) != 0:
                rec = list(record)
                find_index = self.lower_bound(rec,nums[i]-t)
                if find_index != -1 and rec[find_index] <= nums[i] + t:
                    return True
            record.add(nums[i])
            if len(record) == k + 1:
                record.remove(nums[i - k])
        return False
    # 二分查找
    def lower_bound(self, nums, target):
        l, h = 0, len(nums)-1
        while l<h:
            mid = int((l+h)/2)
            if nums[mid] < target:
                l = mid+1
            else:
                h = mid
        return l if nums[l] >= target else -1
```
