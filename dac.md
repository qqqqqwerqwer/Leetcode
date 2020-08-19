# 分治算法
1. 先分 将问题分为若干子问题（不要忘了切分的停止条件啊！）
2. 再治 解决子问题
3. 后合 将子问题解决方案逐次合并

## 示例1 多数元素
问题描述：给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
思路分析：
1. 分 将数组分为左右子数组，直到子数组元素个数为1（停止条件）
2. 治理合并 若左右结果相同，直接返回，不同再整体比较

```python
### 多数元素
class Solution1:
    def majorityElement(self, nums):
        # 分 将数组分为长度为1的子数组
        if not nums:
            return nums
        if len(nums) == 1:
            return nums[0]
        # 治 解决子问题
        left = self.majorityElement(nums[:len(nums)//2])
        right = self.majorityElement(nums[len(nums)//2:])
        # 合
        if left == right:
            return left
        if nums.count(left) > nums.count(right):
            return left
        else:
            return right
```

## 示例2 最大子序和
问题描述：给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
思路分析：
1. 同上
2. 左数组从右到左计算最大子序列和，右数组从左到右计算最大子序列和，最终返回左，右，以及整个数组的最大和中的最大值

```python
### 最大子序和
class Solution2:
    def maxSubArray(self, nums):
        n = len(nums)
        if n == 1:
            return nums[0]
        # 分 为左右子序列
        left = self.maxSubArray(nums[:n//2]) 
        right = self.maxSubArray(nums[n//2:])

        # 治 解决子问题
        max_l = nums[n//2-1]
        tem_l = 0
        for i in range(n//2-1, -1, -1):
            tem_l += nums[i]
            max_l = max(max_l, tem_l)

        max_r = nums[n//2]
        tem_r = 0
        for i in range(n//2, n):
            tem_r += nums[i]
            max_r = max(max_r, tem_r)
        # 合
        return max(left, right, max_l+max_r)
```

## 示例3 pow(x,n)
问题描述：指数计算
思路分析：
1. 指数n不断除以2，直到n=0
2. 通过x乘以x更新x,p = x*self.myPow(x, n-1),最终返回p

```python
### pow(x,n)
class Solution3:
    def myPow(self, x):
        # 特例
        if n < 0:
            n = -n
            x = 1/x
        # 确定终止条件
        if n == 0:
            return 1
        # 治&合
        if n%2 == 1:
            p = x*self.myPow(x, n-1)
            return p
        # 分 n/2
        return self.myPow(x*x, n/2) 
```
