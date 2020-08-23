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

## 示例3 编辑距离
问题描述：给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
示例 1:

输入: word1 = "horse", word2 = "ros"
输出: 3
解释: 
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
     
思路分析：
1. 确定动态规划状态：定义dp[i][j]为字符串word1长度为i和字符串word2长度为j时，word1转化成word2所执行的最少操作次数的值。
   
   转移方程：采用从末尾开始遍历word1和word2， 当word1[i]等于word2[j]时，说明两者完全一样，所以i和j指针可以任何操作都不做，用状态转移式子表示就是dp[i][j]=dp[i-1][j-1]，也就是前一个状态和当前    状态是一样的。
   
   当word1[i]和word2[j]不相等时，就需要对三个操作进行递归了，这里就需要仔细思考状态转移方程的写法了。 对于插入操作，当我们在word1中插入一个和word2一样的字符，那么word2就被匹配    了，所以可    以直接表示为dp[i][j-1]+1 对于删除操作，直接表示为dp[i-1][j]+1 对于替换操作，直接表示为dp[i-1][j-1]+1 所以状态转移方程可以写成min(dp[i][j-1]+1,dp[i-1][j]+1,dp[i-1][j-1]+1)​
   
2. 初始状态：当i=0时，dp[0][j]=j,同理可得，如果另外一个是空字符串，则对当前字符串执行删除操作就可以了，也就是dp[i][0]=i​。
   
   输出状态：最终的编辑距离就是最后一个状态的值，对应的就是dp[-1][-1]​。

3. 优化：由于dp[i][j]只和dp表中附近的三个状态（左边，右边和左上边）有关，所以同样可以进行压缩状态转移的空间存储。

```python
### 编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        # 初始化数组
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        # 处理边界条件，第一列和第一行
        for i in range(n+1):
            dp[0][i] = i
        for j in range(m+1):
            dp[j][0] = j
        # 转移方程
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1] == word2[j-1]: # 比较==，赋值=
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
        # 输出
        return dp[-1][-1]
```

## 示例4 最长上升子序列
问题描述：给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
     
思路分析：
1. 确定动态规划状态：dp[i]可以定义为以nums[i]这个数结尾的最长递增子序列的长度。
   
   转移方程：比较当前dp[i]的长度和dp[i]对应产生新的子序列长度，我们用j来表示所有比i小的组数中的索引
   
2. 初始状态：dp=[1]*len(nums)​
   
   输出状态：max(dp)​

3. 优化：略

```python
### 最长上升子序列
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 特例
        if not nums:
            return 0
        # 初始化数组
        n = len(nums)
        dp = [1]*n
        # 转移方程
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1) # 缩进
        # 输出
        return max(dp)
```

## 示例5 最长连续递增序列
问题描述：给定一个未经排序的整数数组，找到最长且连续的的递增序列。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
     
思路分析：
1. 确定动态规划状态：dp[i]也是以nums[i]这个数结尾的最长连续递增子序列的长度。
   
   转移方程：第一种情况是如果遍历到的数nums[i]后面一个数不是比他大或者前一个数不是比他小，也就是所谓的不是连续的递增，那么这个数列最长连续递增序列就是他本身，也就是长度为1。 第二种情况就是如果    满足有递增序列，就意味着当前状态只和前一个状态有关，dp[i]只需要在前一个状态基础上加一就能得到当前最长连续递增序列的长度。总结起来，状态的转移方程可以写成 dp[i]=dp[i-1]+1
2. 初始状态：dp=[1]*len(nums)​
   
   输出状态：max(dp)​

3. 优化：略
```python
### 最长连续递增序列
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        # 特例
        if not nums:
            return 0
        # 初始化
        n = len(nums)
        dp = [1]*n
        # 转移方程
        for i in range(1,n): # 需要得到前一个数，从nums[1]开始遍历
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1]+1
            else:
                dp[i] = 1
        # 输出
        return max(dp)
```

## 示例6 最长回文子串
问题描述：给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例：

输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
     
思路分析：
1. 确定动态规划状态：定义dp[i][j]表示子串s从i到j是否为回文子串(True or False)
   
   转移方程：字符串首尾两个字符必须相等，否则肯定不是回文。
   
   当字符串首尾两个字符相等时：如果子串是回文，整体就是回文，这里就有了动态规划的思想，出现了子问题；相反，如果子串不是回文，那么整体肯定不是。**(?)**

   对于字符串s,s[i,j]的子串是s[i+1,j-1]，如果子串只有本身或者空串，那肯定是回文子串了，所以我们讨论的状态转移方程不是对于j-1-(i+1)+1<2的情况(整理得j-i<3)，当s[i]和s[j]相等并且j-i<3时，我们    可以直接得出dp[i][j]是True。
2. 初始状态：我们需要建立一个二维的初始状态是False的来保存状态的数组来表示dp，又因为考虑只有一个字符的时候肯定是回文串，所以dp表格的对角线dp[i][i]肯定是True。
   
   输出状态：s[start:start+max_len]

3. 优化：对于这个问题，时间和空间都可以进一步优化，对于空间方面的优化：这里采用一种叫中心扩散的方法来进行，而对于时间方面的优化，则是用了Manacher‘s Algorithm（马拉车算法）来进行优化。

```python
### 最长回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        # 特例
        if n < 2:
            return s
        # 初始化dp
        dp = [[False]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        #初始化max_len,start
        max_len = 1
        start = 0
        # 转移方程
        for j in range(1,n):
            for i in range(j):
                if s[i] == s[j]:
                    if j-i < 2:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]
                # 计算回文子串起始位置和长度
                if dp[i][j]:
                    cur_len = j-i+1
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
        # 输出
        return s[start:start+max_len]
```

## 示例7 最长回文子序列
给定一个字符串s，找到其中最长的回文子序列。可以假设s的最大长度为1000。

示例:
输入:
"bbbab"
输出:
4  **(?)**
     
思路分析：
1. 确定动态规划状态：定义一个二维的dp[i][j]来表示字符串第i个字符到第j个字符的长度，子问题也就是每个子回文字符串的长度。
   
   转移方程：对于d[i][j],我们根据上题的分析依然可以看出， 当s[i]和s[j]相等时，s[i+1...j-1]这个字符串加上2就是最长回文子序列； 当s[i]和s[j]不相等时，就说明可能只有其中一个出现在s[i,j]的最长    回文子序列中，我们只需要取s[i-1,j-1]加上s[i]或者s[j]的数值中较大的；
2. 初始状态：当只有一个字符的时候，最长回文子序列就是1，所以可以得到dp[i][j]=1(i=j) 接下来我们来看看  当i>j时，不符合题目要求，不存在子序列，所以直接初始化为0。 当i<j时，每次计算表中对应的值    就会根据前一个状态的值来计算。
   
   输出状态：dp[0][-1]

3. 优化：对于这个题目，同样可以考虑空间复杂度的优化，因为我们在计算dp[i][j]的时候，只用到左边和下边。如果改为用一维数组存储，那么左边和下边的信息也需要存在数组里，所以我们可以考虑在每次变化前用    临时变量tmp记录会发生变化的左下边信息。

```python
### 最长回文子序列
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        # 特例
        if n < 2:
            return 1
        # 初始化
        dp = [[0]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        # 转移方程
        for i in range(n,-1,-1): #从右下角开始往上遍历
            for j in range(i+1,n):
                if s[i] == s[j]: #当两个字符相等时，直接子字符串加2
                    dp[i][j] = dp[i+1][j-1]+2  
                else: #不相等时，取某边最长的字符
                    dp[i][j] = max(dp[i][j-1],dp[i+1][j])
        # 输出
        return dp[0][-1]
```
