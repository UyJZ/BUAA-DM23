# BUAA-DM23
for 2023-BUAA-DM

## 说明

题目要求见database/作业说明.ppt

## 答案

答案见`database/answer`

## 模型

第一问使用了 https://github.com/cyang-kth/fmm 进行网路匹配

第二问使用了自定义的相似度函数

第三问主要采用了`GRU`

第四问采用了`FPMC`

## 项目说明

`database`包含了各种各样的数据，包括了初始数据，处理后的数据，中间数据等等

`Cluster`主要是第二问聚类相关代码

`ETA`主要是第三问`GRU`的代码 

`FPMC`主要是第四问`FPMC`模型的代码以及后续的预测代码

`GAT-zjy`是`zjy`胎死腹中的图注意力网络代码（最终没有用上）

`roadSchedule`是第三问路径规划的相关代码

## git规范

```bash
# 拉取最新dev分支
git checkout dev
git pull origin

# 签出开发(或修复)分支
git checkout -b feature/xxx (or fix/xxx)

# 提交修改
git add .
git commit -m "[feat](xxx): message" (or "[fix](xxx): message")

# 解决与dev分支的冲突
git checkout dev
git pull origin
git checkout feature/xxx (or fix/xxx)
git rebase dev
git add .
git rebase --continue

# 提交开发/修复分支
git push origin
(pull request on github.com)

# 删除开发/修复分支
git checkout dev
git pull origin
git branch -d feature/xxx (or fix/xxx)
```

