---
name: datanalysis-credit-risk
description: 'Credit risk data cleaning and variable screening pipeline for pre-loan modeling. Use when working with raw credit data that needs quality assessment,  missing value analysis, or variable selection before modeling. it covers data loading and formatting, abnormal period filtering, missing rate calculation, high-missing variable removal,low-IV variable filtering, high-PSI variable removal, Null Importance denoising, high-correlation variable removal, and cleaning report generation. 适用场景：信用风险数据清洗、变量筛选、贷前建模预处理。关键词：缺失率、IV值、PSI、Null Importance、相关性剔除、数据质量报告'
---

# 数据清洗与筛选

## 快速开始

```bash
# 运行完整的数据清洗流程
python ".github/skills/datanalysis-credit-risk/scripts/example.py"
```

## 完整流程说明

数据清洗流程包含以下11个步骤，每个步骤独立执行，不删除原始数据：

1. **获取数据** - 加载并格式化原始数据
2. **机构样本分析** - 统计各机构的样本数量和坏样本率
3. **分离OOS数据** - 将贷外样本与建模样本分离
4. **过滤异常月份** - 剔除坏样本数或总样本数过少的月份
5. **计算缺失率** - 计算各变量的整体缺失率和分机构缺失率
6. **剔除高缺失变量** - 剔除整体缺失率超过阈值的变量
7. **剔除低IV变量** - 剔除整体IV过低或在过多机构上IV过低的变量
8. **剔除高PSI变量** - 剔除PSI不稳定的变量
9. **Null Importance去噪** - 使用标签反转方法剔除噪音变量
10. **剔除高相关变量** - 基于原始gain剔除高相关变量
11. **导出报告** - 生成包含所有步骤详情和统计的Excel报告

## 核心函数

| 函数 | 作用 | 所在模块 |
|------|------|----------|
| `get_dataset()` | 加载并格式化数据 | ai_utils.func |
| `org_analysis()` | 机构样本分析 | ai_utils.func |
| `missing_check()` | 计算缺失率 | ai_utils.func |
| `drop_abnormal_ym()` | 过滤异常月份 | ai_utils.analysis |
| `drop_highmiss_features()` | 剔除高缺失变量 | ai_utils.analysis |
| `drop_lowiv_features()` | 剔除低IV变量 | ai_utils.analysis |
| `drop_highpsi_features()` | 剔除高PSI变量 | ai_utils.analysis |
| `drop_highnoise_features()` | Null Importance去噪 | ai_utils.analysis |
| `drop_highcorr_features()` | 剔除高相关变量 | ai_utils.analysis |
| `iv_distribution_by_org()` | IV分布统计 | ai_utils.analysis |
| `psi_distribution_by_org()` | PSI分布统计 | ai_utils.analysis |
| `value_ratio_distribution_by_org()` | 有值率分布统计 | ai_utils.analysis |
| `export_cleaning_report()` | 导出清洗报告 | ai_utils.analysis |

## 参数说明

### 数据加载参数
- `DATA_PATH`: 数据文件路径（parquet格式）
- `DATE_COL`: 日期列名
- `Y_COL`: 标签列名
- `ORG_COL`: 机构列名
- `KEY_COLS`: 主键列名列表

### OOS机构配置
- `OOS_ORGS`: 贷外机构列表

### 异常月份过滤参数
- `min_ym_bad_sample`: 单月最小坏样本数（默认10）
- `min_ym_sample`: 单月最小总样本数（默认500）

### 缺失率参数
- `missing_ratio`: 整体缺失率阈值（默认0.6）

### IV参数
- `overall_iv_threshold`: 整体IV阈值（默认0.1）
- `org_iv_threshold`: 单机构IV阈值（默认0.1）
- `max_org_threshold`: 最大容忍低IV机构数（默认2）

### PSI参数
- `psi_threshold`: PSI阈值（默认0.1）
- `max_months_ratio`: 最大不稳定月份比例（默认1/3）
- `max_orgs`: 最大不稳定机构数（默认6）

### Null Importance参数
- `n_estimators`: 树的数量（默认100）
- `max_depth`: 树的最大深度（默认5）
- `gain_threshold`: gain差值阈值（默认50）

### 高相关性参数
- `max_corr`: 相关性阈值（默认0.9）
- `top_n_keep`: 保留原始gain排名前N的变量（默认20）

## 输出报告

生成的Excel报告包含以下sheet：

1. **汇总** - 所有步骤的汇总信息，包括操作结果和依据条件
2. **机构样本统计** - 各机构的样本数量和坏样本率
3. **分离OOS数据** - OOS样本和建模样本数量
4. **Step4-异常月份处理** - 被剔除的异常月份
5. **缺失率明细** - 各变量的整体和分机构缺失率
6. **Step5-有值率分布统计** - 各变量在不同有值率区间的分布
7. **Step6-高缺失率处理** - 被剔除的高缺失变量
8. **Step7-IV明细** - 各变量在每个机构上以及整体上的IV值
9. **Step7-IV处理** - 不满足IV条件的变量及低IV机构
10. **Step7-IV分布统计** - 各变量在不同IV区间的分布
11. **Step8-PSI明细** - 各变量在每个机构每个月上的PSI值
12. **Step8-PSI处理** - 不满足PSI条件的变量及不稳定机构
13. **Step8-PSI分布统计** - 各变量在不同PSI区间的分布
14. **Step9-null importance处理** - 被剔除的噪音变量
15. **Step10-高相关性剔除** - 被剔除的高相关变量

## 特点

- **交互式输入**：每个步骤执行前可输入参数，支持默认值
- **独立执行**：各步骤独立执行，不删除原始数据，便于对比分析
- **完整报告**：生成包含明细、统计、分布的完整Excel报告
- **多进程支持**：IV、PSI计算支持多进程加速
- **分机构分析**：支持分机构统计和建模/贷外区分
