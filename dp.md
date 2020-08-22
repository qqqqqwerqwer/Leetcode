# 动态规划
## 应用场景
动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划法仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量，
一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。
## 步骤
1. 确定转移状态和转移方程
2. 确定初始和输出状态
3. 优化
## 示例1 打家劫舍
问题描述：你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

示例:
输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
     
思路分析：
1. 确定动态规划状态：直接定义题目所求的偷窃的最高金额，所以dp[i]表示偷窃第i号房子能得到的最高金额。
   
   转移方程：dp[i]=max(dp[i-2]+nums[i],dp[i-1])​
2. 初始状态：dp[0]=nums[0]，dp[1]=max(nums[0],nums[1])​
   
   输出状态：dp[-1]​
3. 优化：略

```python
### 打家劫舍
class Solution1:
    def rob(self, nums: List[int]) -> int:
        #特例
        if(not nums):
            return 0
        n = len(nums)
        if n == 1:
            return nums[0]
        # 初始化
        dp = [0]*n # 初始化数组
        dp[0] = nums[0] # 第一个边界
        dp[1] = max(nums[0],nums[1]) # 第二个边界
        # 转移方程
        for i in range(2,n):
            dp[i] = max(dp[i - 2]+nums[i],dp[i - 1]) # 调用子问题答案
        # 输出
        return dp[-1]
```

## 示例2 打家劫舍2.0
问题描述：房子组成一个环
     
思路分析：
1. 确定动态规划状态：直接定义题目所求的偷窃的最高金额，所以dp[i]表示偷窃第i号房子能得到的最高金额。
   
   转移方程：dp[i]=max(dp[i-2]+nums[i],dp[i-1])​
   
   偷窃了第一个房子，此时对应的是nums[1:]，得到最大的金额value是v1。
   
   偷窃了最后一个房子，此时对应的是nums[:n-1](其中n是所有房子的数量)，得到的最大金额value是v2。
   
   最后的结果就是取这两种情况的最大值，即max(v1,v2)。
2. 初始状态：dp[0]=nums[0]，dp[1]=max(nums[0],nums[1])​
   
   输出状态：dp[-1]​
3. 优化：略

```python
### 打家劫舍2.0
class Solution2:
    def rob(self, nums: List[int]) -> int:
        #特例
        if not nums:
            return 0
        elif len(nums) <= 2:
            return max(nums)
        # 外面套一个函数
        def helper(nums): #此列表和外面的列表不一样
            if len(nums) <= 2:
                return max(nums)
            # 初始化
            dp = [0]*len(nums) # 初始化数组
            dp[0] = nums[0] # 第一个边界
            dp[1] = max(nums[0],nums[1]) # 第二个边界
            # 转移方程
            for i in range(2,len(nums)):
                dp[i] = max(dp[i - 1],dp[i - 2]+nums[i]) # 调用子问题答案
            # 输出
            return dp[-1]
        return max(helper(nums[1:]),helper(nums[:-1]))
```
