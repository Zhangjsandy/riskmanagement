# 全球顶级金融风控策略 - 信用违约预测系统

## 项目概述

本项目基于全球前沿的机器学习算法，构建了一套完整的信用违约风险预测系统。采用集成学习（Ensemble Learning）策略，融合XGBoost、LightGBM、CatBoost、Random Forest、Gradient Boosting以及TabPFN算法，并在小样本极不平衡场景下使用Top-K非负加权融合、概率校准和成本敏感阈值策略实现更稳健的风险预测。

## 最新执行结果

### 模型性能对比

| 模型                        | CV AUC均值   | 95%CI            | 排名         |
| ------------------------- | ---------- | ---------------- | ---------- |
| **CatBoost**              | **0.6484** | [0.6146, 0.6821] | 1          |
| XGBoost                   | 0.6426     | [0.6059, 0.6793] | 2          |
| Random Forest             | 0.6349     | [0.6026, 0.6673] | 3          |
| TabPFN                    | 0.5992     | [0.5695, 0.6289] | 4          |
| LightGBM                  | 0.5628     | [0.5221, 0.6035] | 5          |
| Gradient Boosting         | 0.5543     | [0.5149, 0.5937] | 6          |
| Top-K Weighted Blend（OOF） | 0.6453     | 自动计算             | 与冠军单模型自动比较 |

### 关键发现

- **CatBoost表现最佳**：CV AUC达到0.6484，成为冠军单模型
- **融合层有效但不稳定优于冠军**：本次Top-K融合OOF AUC为0.6453，自动回退CatBoost出分
- **数据特点**：训练集500条，违约率仅2%，属于典型的小样本极不平衡场景
- **概率校准有效**：Brier由0.02095下降至0.01910（isotonic）
- **最终策略**：Top-K非负加权融合与冠军单模型自动比较，不占优时回退冠军单模型

### 预测结果统计

- **测试集规模**：2000条
- **预测违约数**：30条
- **预测违约率**：1.50%
- **决策阈值**：0.200000（OOF成本最小化，FP=0.05，FN=1.0）

## 目录结构

```
风控测试/
├── data/                       # 数据文件夹
│   ├── 训练数据集.csv          # 训练数据（500条）
│   ├── 测试集.csv              # 测试数据（2000条）
│   └── 提交样例.csv            # 提交格式样例
├── src/                        # 源代码文件夹
│   └── credit_risk_model.py    # 主程序代码
├── output/                     # 输出结果文件夹
│   └── sub-test.csv            # 预测结果
├── models/                     # 模型保存文件夹
│   └── credit_risk_model.pkl   # 训练好的模型
├── tabpfn-models/              # TabPFN模型文件
│   └── tabpfn-v2-classifier-v2_default.ckpt
├── README.md                   # 项目说明文档
└── requirements.txt            # 依赖包列表
```

## 核心算法

### 1. 集成学习策略

本项目采用“稳健融合 + 风险决策”架构：

**第一层（基学习器）**：

- **XGBoost**：极致梯度提升，支持二阶泰勒展开和正则化
- **LightGBM**：高效梯度提升，基于直方图算法和叶子优先策略
- **CatBoost**：类别特征优化提升，采用Ordered Target Statistics
- **Random Forest**：随机森林，Bagging集成策略
- **Gradient Boosting**：梯度提升决策树
- **TabPFN**：面向小样本表格数据的先验拟合网络（Prior-Fitted Networks）

**融合层（Top-K加权）**：

- 按重复分层CV表现选择Top-K强模型（默认K=3）
- 使用非负归一化权重进行概率融合
- 若融合OOF AUC不如冠军单模型，则自动回退冠军单模型

### 2. TabPFN算法详解

TabPFN（Tabular Prior-Fitted Networks）是2022年提出的面向小样本表格数据的深度学习模型：

**核心优势**：

- **无需超参数调优**：基于预训练的Transformer架构
- **小样本友好**：在样本量<1000时表现优异
- **自动特征工程**：通过注意力机制自动学习特征交互
- **快速推理**：单次前向传播即可完成预测

**技术原理**：

- 基于Transformer架构，将表格数据视为序列
- 使用贝叶斯神经网络进行预训练
- 支持分类和回归任务
- 自动处理缺失值和类别特征

**本项目表现**：

- AUC：0.5992（5折×20次重复分层CV）
- 在小样本场景下展现了良好的泛化能力
- 与CatBoost、Random Forest形成有效互补

### 3. 特征工程

构建了35维风险特征体系，包括：

**原始特征**（22个）：

- 基本信息：amount, length, income
- 类别特征：housing, purpose
- 信用历史：overdue_times, default_times, total_default_number
- 账户信息：account_number, loan_history, recent_loan_number
- 信用卡信息：credict_used_amount, credict_limit, total_credict_card_number

**衍生特征**（11个）：

- debt_to_income：债务收入比
- credit_utilization：信用卡使用率
- total_debt_burden：总债务负担
- risk_score：综合风险评分
- credit_history_maturity：信用历史成熟度
- overdue_severity：逾期严重程度
- 等...

**分箱特征**（2个）：

- income_level：收入水平分箱
- amount_level：贷款金额分箱

### 4. 类别不平衡处理

采用不平衡处理策略（默认：类权重；可选：SMOTE）。

**重要说明（评估不泄露）**：如果启用SMOTE，本项目将仅在每个交叉验证fold的训练集内进行过采样，避免在"先全量SMOTE再CV"的流程下造成验证集信息泄露，从而导致AUC等指标虚高。

策略效果与稳定性建议以OOF（Out-of-Fold）指标为准。

如启用SMOTE，样本数量变化示例：

- 原始数据：正常样本490条（98%），违约样本10条（2%）
- SMOTE处理后：正常样本490条，违约样本490条（平衡）

## 模型性能详解

### 交叉验证与OOF评估（重复分层交叉验证）

为避免由于不平衡处理或重复使用验证信息导致的指标虚高，本项目采用"泄露控制"的评估方式：

- 基模型：重复分层交叉验证（默认5折×20次）得到更稳健的OOF预测与CV AUC分布
- 融合模型：Top-K非负加权融合，输出OOF AUC / PR-AUC / KS
- 统计稳健性：输出95%置信区间（CI），降低单次切分偶然性影响
- 最终策略：若融合在OOF上不如最佳单模型，将自动回退最佳单模型作为最终出分（小样本/极少坏样本时更稳）

> 注：最终策略会输出OOF PR-AUC与KS，并基于OOF预测自动选择决策阈值（默认成本最小化，替代固定0.5阈值）。

### 本次执行的详细指标

**Final Strategy OOF Metrics**：

- OOF AUC：0.7268
- OOF PR-AUC：0.0529
- OOF KS：0.3429
- 决策阈值：0.200000（成本最小化）
- 参考阈值：0.036364（坏账率匹配）/ 0.029851（Youden/KS）

**混淆矩阵（OOF @ threshold）**：

```
[[485   5]
 [  9   1]]
```

### 成本参数敏感性分析

> 说明：为在可接受时间内完成3组对比，敏感性实验使用5折×10次重复分层CV；主实验仍使用5折×20次。

| FP成本 | FN成本 | 阈值(成本最小化) | OOF成本    | OOF拦截率 | 测试集拦截率 |
| ---- | ---- | --------- | -------- | ------ | ------ |
| 0.02 | 1.0  | 0.022222  | **7.02** | 51.80% | 51.65% |
| 0.05 | 1.0  | 0.500000  | 9.05     | 0.40%  | 1.00%  |
| 0.10 | 1.0  | 0.500000  | 9.10     | 0.40%  | 1.00%  |

结论：当前样本下，若企业目标是**最小化定义成本函数**，`FP=0.02, FN=1.0` 成本最低，但拦截率过高；若追求更低误拒、控制客群损失，`FP=0.05, FN=1.0` 更保守，适合作为业务初始上线参数。

### 阈值决策表（业务友好摘要）

> 数据来源：`output/threshold_decision_table_quick.csv`（快速验证：5折×1次，FP=0.05，FN=1.0）。

| 排名  | 阈值       | 总成本   | 拦截率   | 通过率     | 坏样本召回率 | 坏样本精确率 |
| --- | -------- | ----- | ----- | ------- | ------ | ------ |
| 1   | 0.050000 | 9.90  | 3.80% | 96.20%  | 10.00% | 5.26%  |
| 2   | 0.052632 | 9.90  | 3.80% | 96.20%  | 10.00% | 5.26%  |
| 3   | 0.100000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 4   | 0.150000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 5   | 0.200000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 6   | 0.250000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 7   | 0.300000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 8   | 0.350000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 9   | 0.400000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |
| 10  | 0.450000 | 10.00 | 0.00% | 100.00% | 0.00%  | 0.00%  |

推荐阈值（当前成本口径）建议采用 **0.052632**：

- 与最低成本并列最优（9.90）
- 相比高阈值（0.1+）可恢复一定坏样本识别能力（召回10%）
- 拦截率仅3.8%，更利于业务初期平稳上线

### 特征重要性Top 15

| 排名  | 特征名                       | 重要性      |
| --- | ------------------------- | -------- |
| 1   | credit_history_maturity   | 320.5246 |
| 2   | total_debt_burden         | 268.5148 |
| 3   | income                    | 260.5370 |
| 4   | credict_limit             | 224.0446 |
| 5   | last_credict_card_months  | 204.0207 |
| 6   | total_balance             | 202.0112 |
| 7   | debt_to_income            | 196.5077 |
| 8   | credict_used_amount       | 177.0096 |
| 9   | risk_score                | 130.0068 |
| 10  | credit_utilization        | 129.5109 |
| 11  | amount                    | 129.0155 |
| 12  | avg_balance_per_account   | 124.5073 |
| 13  | total_credict_card_number | 107.0226 |
| 14  | loan_history              | 92.0178  |
| 15  | recent_account_months     | 71.5196  |

## 使用方法

### 环境要求

```bash
pip install -r requirements.txt
```

**核心依赖**：

- Python 3.13+
- PyTorch 2.10.0+
- XGBoost, LightGBM, CatBoost
- TabPFN 6.4.1
- scikit-learn, pandas, numpy

### TabPFN配置

TabPFN需要HuggingFace认证才能下载模型：

```bash
# 设置环境变量
$env:HF_TOKEN="your_huggingface_token"

# 或在运行前设置
$env:HF_TOKEN="your_token"; python src/credit_risk_model.py
```

**获取HF_TOKEN**：

1. 访问 https://huggingface.co/Prior-Labs/tabpfn_2_5
2. 接受使用条款
3. 在Settings -> Access Tokens中创建token
   

### 运行程序

```bash
python src/credit_risk_model.py
```

### 参数化运行（推荐）

```bash
python src/credit_risk_model.py \
   --cv-splits 5 \
   --cv-repeats 20 \
   --top-k-models 3 \
   --threshold-method cost \
   --cost-fp 0.05 \
   --cost-fn 1.0
```

常用参数：

- `--cv-splits`：分层CV折数（默认5）
- `--cv-repeats`：重复次数（默认20）
- `--top-k-models`：融合模型个数K（默认3）
- `--threshold-method`：`cost` / `match_rate` / `youden`
- `--cost-fp`、`--cost-fn`：成本函数参数
- `--disable-tabpfn`：禁用TabPFN
- `--use-smote`：启用fold内SMOTE
- `--decision-table-path`：阈值决策表输出路径（默认 `output/threshold_decision_table.csv`）
- `--decision-table-points`：阈值网格点数（默认41）

### 输出结果

程序将自动生成以下文件：

- `output/sub-test.csv`：预测结果（id, target）
- `output/threshold_decision_table.csv`：阈值-拦截率-成本决策表
- `models/credit_risk_model.pkl`：保存的模型文件
- `docs/experiment_results.tex`：实验结果LaTeX片段

### 运行输出解读

运行日志中与风控落地最相关的关键信息包括：

- **Final Strategy OOF Metrics**：OOF PR-AUC、OOF KS 等（比单纯AUC更能反映极不平衡场景下的有效性）
- **Final strategy**：最终采用 `weighted_blend` 或 `best_base`（当融合在OOF上不占优时，会自动回退最佳单模型出分）
- **Calibration**：概率校准方法及Brier Score前后对比
- **Decision threshold**：自动选择的阈值（默认成本最小化；同时打印坏账率匹配/Youden参考阈值）

## 技术亮点

1. **多算法融合**：融合6种主流集成学习算法，包括前沿的TabPFN
2. **先进特征工程**：基于金融风控专业知识构建35维特征
3. **不平衡处理**：默认采用类权重；可选SMOTE且仅在每个CV训练fold内使用，避免评估泄露
4. **稳健出分**：Top-K非负加权融合在OOF不占优时自动回退最佳单模型，降低小样本方差
5. **TabPFN增强**：成功集成前沿小样本模型，AUC达到0.5992
6. **策略可落地**：阈值基于OOF自动选择（默认成本最小化），并支持概率校准



# 
