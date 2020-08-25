# 查找表
## 考虑基本数据结构
第一类： 查找有无--set

元素'a'是否存在，通常用set：集合

set只存储键，而不需要对应其相应的值。

set中的键不允许重复

第二类： 查找对应关系(键值对应)--dict

元素'a'出现了几次：dict-->字典

dict中的键不允许重复

第三类： 改变映射关系--map

通过将原有序列的关系映射统一表示为其他

## 示例1 两个数组的交集
问题描述：给定两个数组nums,求两个数组的公共元素。

如nums1 = [1,2,2,1],nums2 = [2,2]

结果为[2]
结果中每个元素只能出现一次
出现的顺序可以是任意的
     
思路分析：
元素是否存在用set

```python
### 两个数组的交集set
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        return set(i for i in nums1 if i in nums2)
```

```python
### 两个数组的交集&
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        nums2 = set(nums2)
        return nums1 & nums2
```

## 示例2 两个数组的交集2.0
问题描述：给定两个数组nums,求两个数组的交集。

-- 如nums1=[1,2,2,1],nums=[2,2]

-- 结果为[2,2]

-- 出现的顺序可以是任意的
     
思路分析：
由于每个元素出现的次数有影响，故选用字典dict。

```python
### 两个数组的交集set
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        from collections import Counter
        dic_nums1 = Counter(nums1)
        res = []
        for i in nums2:
            if dic_nums1[i] > 0:
                res.append(i)
                dic_nums1[i] -= 1
        return res
```

## 示例3 有效的字母异位词
问题描述：给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

示例1:

输入: s = "anagram", t = "nagaram"
输出: true

示例 2:

输入: s = "rat", t = "car"
输出: false
     
思路分析：
不仅需要存储元素，还需要记录元素的个数。可以选择dict的数据结构，将字符串s和t都用dict存储，而后直接比较两个dict是否相同。

```python
### 有效的字母异位词
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        dict_1 = Counter(s)
        dict_2 = Counter(t)
        if dict_1 == dict_2:
            return True
        else:
            return False
```

## 示例3 happy number
问题描述：编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

示例: 
输入: 19
输出: true
解释: 
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
     
思路分析：
当n不等于1时就循环，每次循环时，将其最后一位到第一位的数依次平方求和，比较求和是否为1。

因为只需要判断有或无，不需要记录次数，故用set的数据结构。每次对求和的数进行append，当新一次求和的值存在于set中时，就return false.

```python
### happy number
class Solution:
    def isHappy(self, n: int) -> bool:
        already = set()
        while n != 1: # while循环
            s = 0
            while n > 0: # while循环
                tem = n%10
                s += tem*tem
                n //= 10
            if s in already:
                return False
            else:
                already.add(s) # 集合添加元素函数add
            n = s
        return True
```

## 示例4 word pattern
问题描述：给出一个模式(pattern)以及一个字符串，判断这个字符串是否符合模式

示例1:
输入: pattern = "abba", 
str = "dog cat cat dog"
输出: true

示例 2:
输入:pattern = "abba", 
str = "dog cat cat fish"
输出: false

示例 3:
输入: pattern = "aaaa", str = "dog cat cat dog"
输出: false

示例 4:
输入: pattern = "abba", str = "dog dog dog dog"
输出: false
     
思路分析：
将原来的dict通过map映射为相同的key，再比较相同key的dict是否相同。

```python
### word pattern
class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
        str = str.split() # 根据空格拆成字符list
        return list(map(pattern.index,pattern)) == list(map(str.index,str)) # 通过map将字典映射为index的list,map是通过hash存储的，不能直接进行比较，需要转换为list比较list
```

## 示例5 同构字符串
问题描述：给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。

所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

示例 1:
输入: s = "egg", t = "add"
输出: true

示例 2:
输入: s = "foo", t = "bar"
输出: false

示例 3:
输入: s = "paper", t = "title"
输出: true
     
思路分析：
同上

```python
### 同构字符串
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return list(map(s.index,s)) == list(map(t.index,t)) # map函数
```

## 示例6 根据字符出现的频率排序
问题描述：给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

示例 1:
输入:
"tree"
输出:
"eert"

示例 2:
输入:
"cccaaa"
输出:
"cccaaa"

示例 3:
输入:
"Aabb"
输出:
"bbAa"
     
思路分析：
使用字典统计频率，对字典的value进行排序，最终根据key的字符串乘上value次数，组合在一起输出。

```python
### 根据字符出现的频率排序
class Solution:
    def frequencySort(self, s: str) -> str:
        from collections import Counter
        s_dict = Counter(s)
        s = sorted(s_dict.items(), key = lambda item:item[1], reverse = True)
        res = ''
        for key,value in s:
            res += key*value
        return res
```

## 示例7 搜索插入位置
问题描述：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

示例:

输入: [1,3,5,6], 5
输出: 2

思路分析：
二分法

```python
### 搜索插入位置
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        if n == 0:
            return 0
        if nums[0] > target:
            return 0    
        if nums[n-1] < target:
            return n
        
        left = 0
        right = n-1

        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
```

## 示例8 有序数组中的单一元素
问题描述：给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

示例 1:

输入: [1,1,2,3,3,4,4,8,8]
输出: 2

示例 2:

输入: [3,3,7,7,10,11,11]
输出: 10

思路分析：
二分法

```python
### 有序数组中的单一元素
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left + right) // 2
            if mid % 2 == 0 and mid + 1 < len(nums): 
                if nums[mid] == nums[mid+1]:
                    left = mid + 1
                else:
                    right = mid - 1
            elif mid % 2 != 0 and mid + 1 < len(nums):
                if nums[mid] == nums[mid+1]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                return nums[mid]
        return nums[left]
```
