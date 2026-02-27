# -*- coding: utf-8 -*-
"""
================================================================================
全球顶级金融风控策略 - 信用违约预测模型
基于最新集成学习算法：XGBoost + LightGBM + CatBoost + Weighted Blend Ensemble
================================================================================


Description:
    本模型采用全球最先进的金融风险预测算法，包括：
    1. XGBoost - 极致梯度提升
    2. LightGBM - 高效梯度提升
    3. CatBoost - 类别特征优化提升
    4. Top-K Weighted Blend - 模型融合技术
    5. SMOTE - 不平衡数据处理
    6. 特征工程优化
"""

import pandas as pd
import numpy as np
import warnings
import copy
import argparse
warnings.filterwarnings('ignore')

# 数据预处理
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

# 高级模型
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    TABPFN_IMPORT_ERROR = None
    # 尝试获取TabPFN版本以确定API
    import tabpfn
    TABPFN_VERSION = getattr(tabpfn, '__version__', 'unknown')
except Exception as e:
    TabPFNClassifier = None
    TABPFN_AVAILABLE = False
    TABPFN_IMPORT_ERROR = str(e)
    TABPFN_VERSION = None

# 不平衡数据处理
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imblearn not installed. SMOTE will not be used.")

# 模型保存
import joblib
import os
from datetime import datetime

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CreditRiskModel:
    """
    信用风险预测模型类

    集成多种先进算法进行违约预测，包括XGBoost、LightGBM、CatBoost等
    基础模型，并支持Top-K加权融合、概率校准、阈值优化等功能。

    Attributes:
        label_encoders (dict): 类别特征的标签编码器字典
        scaler (StandardScaler): 数值特征标准化器
        models (dict): 训练后的基础模型字典
        meta_model (LogisticRegression): 元模型（Stacking第二层）
        feature_importance (dict): 特征重要性字典
        categorical_features (list): 类别特征名称列表
        numeric_features (list): 数值特征名称列表
        decision_threshold (float): 决策阈值
        cv_splits (int): 交叉验证折数
        cv_repeats (int): 交叉验证重复次数
        top_k_models (int): Top-K融合模型数量
    """

    def __init__(self):
        """初始化信用风险模型，设置默认参数。"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.meta_model = None
        self.feature_importance = None
        self.categorical_features = ['housing', 'purpose']
        self.numeric_features = None
        self.decision_threshold = 0.5
        self.oof_meta_pred = None
        self.meta_cv_scores = None
        self.blend_metrics = None
        self.final_oof_metrics = None
        self.base_model_names = []
        self.selected_base_models = []
        self.blend_weights = None
        self.final_strategy = 'weighted_blend'  # 'weighted_blend' or 'best_base'
        self.champion_model_name = None
        self.oof_final_pred = None
        self.oof_final_pred_calibrated = None
        self.last_test_stats = None
        self.use_tabpfn = False
        self.tabpfn_requested = False
        self.tabpfn_ready = None
        self.tabpfn_unavailable_reason = None
        self.calibrator = None
        self.calibration_method = 'isotonic'
        self.threshold_method = 'cost'
        self.cost_fp = 0.05  # 好客户误拒绝成本
        self.cost_fn = 1.00  # 坏客户误通过成本
        self.cv_splits = 5
        self.cv_repeats = 20
        self.top_k_models = 3

    def load_data(self, train_path, test_path):
        """
        加载训练和测试数据

        Args:
            train_path (str): 训练数据文件路径
            test_path (str): 测试数据文件路径

        Returns:
            tuple: (train_df, test_df) 训练集和测试集的DataFrame
        """
        print("=" * 80)
        print("Loading datasets...")
        print("=" * 80)

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"Training set shape: {self.train_df.shape}")
        print(f"Test set shape: {self.test_df.shape}")
        print(f"\nTraining set columns: {list(self.train_df.columns)}")
        print(f"\nTarget distribution:\n{self.train_df['target'].value_counts()}")
        print(f"Default rate: {self.train_df['target'].mean():.4f}")

        return self.train_df, self.test_df

    def feature_engineering(self, df, is_train=True):
        """
        特征工程 - 创建高级特征

        基于原始特征构建新的衍生特征，包括债务收入比、信用卡使用率、
        历史违约率、风险评分等多个维度。

        Args:
            df (pd.DataFrame): 输入数据框
            is_train (bool): 是否为训练数据，默认为True

        Returns:
            pd.DataFrame: 添加了新特征的数据框
        """
        df = df.copy()

        # 1. 债务收入比特征
        df['debt_to_income'] = df['amount'] / (df['income'] + 1)

        # 2. 信用卡使用率
        df['credit_utilization'] = df['credict_used_amount'] / (df['credict_limit'] + 1)

        # 3. 平均账户余额
        df['avg_balance_per_account'] = df['total_balance'] / (df['account_number'] + 1)

        # 4. 历史违约率
        df['default_rate_history'] = df['default_times'] / (df['loan_history'] + 1)

        # 5. 近期活跃程度
        df['recent_activity_ratio'] = df['recent_loan_number'] / (df['loan_history'] + 1)

        # 6. 信用卡活跃率
        df['credit_card_activity'] = df['half_used_credict_card'] / (df['total_credict_card_number'] + 1)

        # 7. 查询频率
        df['inquiry_frequency'] = df['inquire_times'] / (df['recent_account_months'] + 1)

        # 8. 总负债水平
        df['total_debt_burden'] = (df['mortage_number'] + df['account_number']) / (df['income'] / 1000 + 1)

        # 9. 信用历史长度与年龄比
        df['credit_history_maturity'] = df['last_credict_card_months'] / (df['length'] + 1)

        # 10. 逾期严重程度
        df['overdue_severity'] = df['overdue_times'] * df['last_overdue_months']

        # 11. 多维度风险评分
        df['risk_score'] = (
            df['overdue_times'] * 0.3 +
            df['default_times'] * 0.4 +
            df['inquire_times'] * 0.1 +
            (1 - df['credit_utilization']) * 0.2
        )

        # 12. 收入分箱
        df['income_level'] = pd.cut(df['income'],
                                     bins=[0, 3000, 6000, 10000, 20000, float('inf')],
                                     labels=['very_low', 'low', 'medium', 'high', 'very_high'])

        # 13. 贷款金额分箱
        df['amount_level'] = pd.cut(df['amount'],
                                     bins=[0, 5000, 10000, 20000, float('inf')],
                                     labels=['small', 'medium', 'large', 'very_large'])

        return df

    def preprocess(self, df, is_train=True):
        """
        数据预处理

        包括特征工程、类别特征编码、缺失值处理等步骤。

        Args:
            df (pd.DataFrame): 输入数据框
            is_train (bool): 是否为训练数据，默认为True

        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        df = df.copy()

        # 特征工程
        df = self.feature_engineering(df, is_train)

        # 处理类别特征
        for col in self.categorical_features + ['income_level', 'amount_level']:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # 处理未见过的类别
                        df[col] = df[col].astype(str)
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[col] = le.transform(df[col])

        # 处理缺失值
        df = df.fillna(df.median(numeric_only=True))

        return df

    def prepare_features(self, df, is_train=True):
        """
        准备特征矩阵

        从数据框中提取特征列和目标列。

        Args:
            df (pd.DataFrame): 输入数据框
            is_train (bool): 是否为训练数据，默认为True

        Returns:
            tuple: 若is_train为True返回(X, y, feature_cols)，否则返回(X, feature_cols)
        """
        # 排除ID和目标列
        exclude_cols = ['id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]

        if is_train:
            y = df['target']
            return X, y, feature_cols
        else:
            return X, feature_cols

    def handle_imbalance(self, X, y, verbose=True):
        """
        处理类别不平衡问题

        使用SMOTE算法对少数类进行过采样。

        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            verbose (bool): 是否打印详细信息，默认为True

        Returns:
            tuple: (X_resampled, y_resampled) 平衡后的特征和目标
        """
        if not IMBLEARN_AVAILABLE:
            if verbose:
                print("imblearn not available, skipping SMOTE...")
            return X, y

        if verbose:
            print("\nApplying SMOTE for class imbalance...")
            print(f"Before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")

        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)

        if verbose:
            print(f"After SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

        return X_resampled, y_resampled

    def _ks_stat(self, y_true, y_prob):
        """
        计算KS统计量

        Args:
            y_true (array-like): 真实标签
            y_prob (array-like): 预测概率

        Returns:
            float: KS统计量值
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float(np.max(tpr - fpr))

    def _find_optimal_threshold(self, y_true, y_prob, method='youden'):
        """
        在训练集OOF预测上选择最优阈值

        Args:
            y_true (array-like): 真实标签
            y_prob (array-like): 预测概率
            method (str): 阈值选择方法，可选'youden'、'match_rate'、'cost'

        Returns:
            float: 最优阈值
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        if method == 'youden':
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            return float(thresholds[best_idx])

        if method == 'match_rate':
            # 让预测为1的比例接近训练集坏样本率
            target_rate = float(np.mean(y_true))
            target_rate = min(max(target_rate, 1e-6), 1 - 1e-6)
            return float(np.quantile(y_prob, 1 - target_rate))

        if method == 'cost':
            thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
            if len(thresholds) == 0:
                return 0.5

            best_thr = 0.5
            best_cost = float('inf')
            for thr in thresholds:
                y_hat = (y_prob >= thr).astype(int)
                fp = int(np.sum((y_hat == 1) & (y_true == 0)))
                fn = int(np.sum((y_hat == 0) & (y_true == 1)))
                cost = self.cost_fp * fp + self.cost_fn * fn
                if cost < best_cost:
                    best_cost = cost
                    best_thr = float(thr)

            return best_thr

        return 0.5

    def _confidence_interval_95(self, scores):
        """
        计算95%置信区间

        Args:
            scores (array-like): 分数数组

        Returns:
            tuple: (ci_low, ci_high) 置信区间下界和上界
        """
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return (float('nan'), float('nan'))
        if scores.size == 1:
            return (float(scores[0]), float(scores[0]))

        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        se = std / np.sqrt(scores.size)
        margin = 1.96 * se
        return (mean - margin, mean + margin)

    def _fit_probability_calibrator(self, y_true, y_prob, method='isotonic'):
        """
        拟合概率校准器

        Args:
            y_true (array-like): 真实标签
            y_prob (array-like): 预测概率
            method (str): 校准方法，可选'isotonic'或'sigmoid'

        Returns:
            dict: 包含校准方法和模型对象的字典
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if method == 'isotonic':
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(y_prob, y_true)
            return {'method': 'isotonic', 'model': cal}

        if method == 'sigmoid':
            cal = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
            cal.fit(y_prob.reshape(-1, 1), y_true)
            return {'method': 'sigmoid', 'model': cal}

        return None

    def _apply_probability_calibration(self, y_prob):
        """
        应用概率校准

        Args:
            y_prob (array-like): 原始预测概率

        Returns:
            array: 校准后的概率
        """
        y_prob = np.asarray(y_prob)
        if self.calibrator is None:
            return y_prob

        method = self.calibrator.get('method')
        model = self.calibrator.get('model')
        if method == 'isotonic':
            return np.asarray(model.transform(y_prob), dtype=float)
        if method == 'sigmoid':
            return np.asarray(model.predict_proba(y_prob.reshape(-1, 1))[:, 1], dtype=float)
        return y_prob

    def _fit_with_optional_weights(self, model, X, y, use_class_weight=True):
        """
        使用可选的样本权重拟合模型

        Args:
            model: 待拟合的模型对象
            X (array-like): 特征矩阵
            y (array-like): 目标变量
            use_class_weight (bool): 是否使用类别权重，默认为True

        Returns:
            object: 拟合后的模型对象
        """
        sample_weight = None
        if use_class_weight:
            try:
                sample_weight = compute_sample_weight(class_weight='balanced', y=y)
            except Exception:
                sample_weight = None

        if sample_weight is not None:
            try:
                model.fit(X, y, sample_weight=sample_weight)
                return model
            except Exception:
                pass

        model.fit(X, y)
        return model

    def _clone_model(self, model):
        """
        克隆模型

        Args:
            model: 待克隆的模型对象

        Returns:
            object: 克隆后的模型对象
        """
        try:
            return clone(model)
        except Exception:
            return copy.deepcopy(model)

    def _check_tabpfn_ready(self):
        """
        检查TabPFN模型是否可用

        Returns:
            tuple: (is_ready, error_message) 是否可用及错误信息
        """
        if not TABPFN_AVAILABLE:
            return False, f"import failed: {TABPFN_IMPORT_ERROR}"

        try:
            X_chk = np.array(
                [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]],
                dtype=float,
            )
            y_chk = np.array([0, 1, 0, 1], dtype=int)
            clf = TabPFNClassifier()
            clf.fit(X_chk, y_chk)
            _ = clf.predict_proba(X_chk)
            return True, None
        except Exception as e:
            error_msg = str(e)
            if any(kw in error_msg.lower() for kw in ['connect', 'timeout', 'network', 'download', 'hub', 'internet']):
                return False, f"network error (model download failed): {error_msg[:100]}"
            return False, error_msg

    def _latex_escape(self, text):
        """
        转义LaTeX特殊字符

        Args:
            text (str): 原始文本

        Returns:
            str: 转义后的文本
        """
        if text is None:
            return ''
        return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_')

    def create_base_models(self):
        """
        创建基础模型集合

        创建包括XGBoost、LightGBM、CatBoost、RandomForest、GradientBoosting
        在内的基础分类模型。

        Returns:
            dict: 模型名称到模型对象的映射字典
        """
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='auc',
                use_label_encoder=False,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=RANDOM_STATE,
                verbose=False,
                loss_function='Logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            )
        }

        if self.use_tabpfn:
            if self.tabpfn_ready is None:
                self.tabpfn_ready, self.tabpfn_unavailable_reason = self._check_tabpfn_ready()

            if self.tabpfn_ready:
                models['tabpfn'] = TabPFNClassifier()
                print("TabPFN enabled as a base model.")
            else:
                print(f"TabPFN disabled (reason: {self.tabpfn_unavailable_reason})")

        return models

    def train_base_models(self, X, y, n_splits=5, n_repeats=20, use_smote=False, use_class_weight=True):
        """
        训练基础模型并进行交叉验证

        重要：若使用SMOTE，仅在每个fold的训练集内进行，避免数据泄露导致CV分数虚高。

        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            n_splits (int): 交叉验证折数，默认为5
            n_repeats (int): 交叉验证重复次数，默认为20
            use_smote (bool): 是否使用SMOTE，默认为False
            use_class_weight (bool): 是否使用类别权重，默认为True

        Returns:
            tuple: (cv_scores, oof_predictions) 交叉验证分数和OOF预测
        """
        print("\n" + "=" * 80)
        print("Training Base Models")
        print("=" * 80)

        self.models = self.create_base_models()
        if len(self.models) == 0:
            raise RuntimeError("No base models available for training.")
        self.base_model_names = list(self.models.keys())
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )
        split_indices = list(cv.split(X, y))

        oof_sum = np.zeros((len(X), len(self.models)), dtype=float)
        oof_count = np.zeros((len(X), len(self.models)), dtype=float)
        cv_scores = {}

        for idx, (name, base_model) in enumerate(self.models.items()):
            print(f"\nTraining {name}...")

            scores = []
            for fold, (train_idx, val_idx) in enumerate(split_indices, start=1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = self._clone_model(base_model)

                if use_smote and IMBLEARN_AVAILABLE:
                    X_fit, y_fit = self.handle_imbalance(X_train, y_train, verbose=False)
                    model = self._fit_with_optional_weights(model, X_fit, y_fit, use_class_weight=False)
                else:
                    model = self._fit_with_optional_weights(model, X_train, y_train, use_class_weight=use_class_weight)

                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)

                auc = roc_auc_score(y_val, y_pred)
                scores.append(auc)
                oof_sum[val_idx, idx] += y_pred
                oof_count[val_idx, idx] += 1.0

            # 重新训练完整数据（同策略）
            final_model = self._clone_model(base_model)
            if use_smote and IMBLEARN_AVAILABLE:
                X_fit, y_fit = self.handle_imbalance(X, y, verbose=True)
                final_model = self._fit_with_optional_weights(final_model, X_fit, y_fit, use_class_weight=False)
            else:
                final_model = self._fit_with_optional_weights(final_model, X, y, use_class_weight=use_class_weight)

            self.models[name] = final_model

            ci_low, ci_high = self._confidence_interval_95(scores)
            cv_scores[name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'ci95_low': ci_low,
                'ci95_high': ci_high,
                'n_folds': len(scores),
                'scores': scores
            }

            print(
                f"  CV AUC: {np.mean(scores):.4f} (95% CI: {ci_low:.4f}, {ci_high:.4f}; "
                f"folds={len(scores)})"
            )

        oof_predictions = np.divide(
            oof_sum,
            np.maximum(oof_count, 1e-12),
            out=np.zeros_like(oof_sum),
            where=oof_count > 0,
        )

        self.oof_predictions = oof_predictions
        self.oof_prediction_counts = oof_count
        self.cv_scores = cv_scores

        return cv_scores, oof_predictions

    def build_weighted_blend(self, oof_predictions, cv_scores, top_k=3):
        """
        构建Top-K加权融合模型

        选择表现最好的top_k个模型，根据AUC表现计算权重进行融合。

        Args:
            oof_predictions (np.ndarray): OOF预测结果矩阵
            cv_scores (dict): 各模型的交叉验证分数
            top_k (int): 选择前k个模型，默认为3

        Returns:
            tuple: (blend_oof, blend_metrics) 融合后的OOF预测和评估指标
        """
        ranked = sorted(cv_scores.items(), key=lambda kv: kv[1]['mean'], reverse=True)
        selected = [name for name, _ in ranked[:top_k]]
        if len(selected) == 0:
            raise RuntimeError('No base models available for weighted blend.')

        indices = [self.base_model_names.index(name) for name in selected]
        raw_strength = np.array([max(cv_scores[name]['mean'] - 0.5, 0.0) for name in selected], dtype=float)
        if float(np.sum(raw_strength)) <= 0:
            raw_strength = np.ones_like(raw_strength)
        weights = raw_strength / float(np.sum(raw_strength))

        blend_oof = np.dot(oof_predictions[:, indices], weights)
        metrics = {
            'auc': float(roc_auc_score(np.asarray(self.y_train), blend_oof)),
            'pr_auc': float(average_precision_score(np.asarray(self.y_train), blend_oof)),
            'ks': float(self._ks_stat(np.asarray(self.y_train), blend_oof)),
            'selected_models': selected,
            'weights': weights.tolist(),
        }
        return blend_oof, metrics

    def train_meta_model(self, oof_predictions, y, cv=5):
        """
        训练元模型（Stacking第二层）

        使用第二层CV对元模型做OOF评估，避免把"用同一份OOF特征训练出来的元模型"
        直接在同一份数据上评估导致的乐观偏差。

        Args:
            oof_predictions (np.ndarray): 基础模型的OOF预测
            y (array-like): 目标变量
            cv (int): 交叉验证折数，默认为5

        Returns:
            float: 元模型的平均AUC分数
        """
        print("\n" + "=" * 80)
        print("Training Meta Model (Stacking)")
        print("=" * 80)

        y_arr = np.asarray(y)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        oof_meta = np.zeros(len(y_arr), dtype=float)
        meta_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(oof_predictions, y_arr)):
            model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
            model.fit(oof_predictions[train_idx], y_arr[train_idx])
            oof_meta[val_idx] = model.predict_proba(oof_predictions[val_idx])[:, 1]
            meta_scores.append(roc_auc_score(y_arr[val_idx], oof_meta[val_idx]))

        meta_auc_mean = float(np.mean(meta_scores))
        meta_auc_std = float(np.std(meta_scores))
        meta_ap = float(average_precision_score(y_arr, oof_meta))
        meta_ks = self._ks_stat(y_arr, oof_meta)

        print(f"Meta Model CV AUC: {meta_auc_mean:.4f} (+/- {meta_auc_std * 2:.4f})")
        print(f"Meta Model OOF PR-AUC: {meta_ap:.4f}")
        print(f"Meta Model OOF KS: {meta_ks:.4f}")

        # 训练最终元模型
        self.meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        self.meta_model.fit(oof_predictions, y_arr)

        self.oof_meta_pred = oof_meta
        self.meta_cv_scores = {
            'auc_mean': meta_auc_mean,
            'auc_std': meta_auc_std,
            'pr_auc': meta_ap,
            'ks': meta_ks,
            'fold_auc': meta_scores,
        }

        return meta_auc_mean

    def train(self, train_path, test_path, use_smote=False, use_tabpfn=True):
        """
        完整训练流程

        执行从数据加载到模型训练的完整流程。

        Args:
            train_path (str): 训练数据路径
            test_path (str): 测试数据路径
            use_smote (bool): 是否使用SMOTE，默认为False
            use_tabpfn (bool): 是否使用TabPFN，默认为True

        Returns:
            tuple: (cv_scores, final_auc) 交叉验证分数和最终AUC
        """
        self.use_tabpfn = bool(use_tabpfn)
        self.tabpfn_requested = bool(use_tabpfn)
        self.tabpfn_ready = None
        self.tabpfn_unavailable_reason = None

        # 加载数据
        self.load_data(train_path, test_path)

        # 预处理
        print("\n" + "=" * 80)
        print("Preprocessing Data")
        print("=" * 80)

        train_processed = self.preprocess(self.train_df, is_train=True)
        test_processed = self.preprocess(self.test_df, is_train=False)

        # 准备特征
        X_train, y_train, self.feature_cols = self.prepare_features(train_processed, is_train=True)
        X_test, _ = self.prepare_features(test_processed, is_train=False)

        # 供训练后汇总与阈值/策略分析使用
        self.X_train = X_train
        self.y_train = y_train

        print(f"\nFeature count: {len(self.feature_cols)}")
        print(f"Features: {self.feature_cols}")

        # 训练基础模型
        cv_scores, oof_predictions = self.train_base_models(
            X_train,
            y_train,
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            use_smote=use_smote,
            use_class_weight=(not use_smote),
        )

        # 选出最佳单模型
        best_base_name = max(cv_scores.items(), key=lambda kv: kv[1]['mean'])[0]
        best_base_auc = float(cv_scores[best_base_name]['mean'])
        best_base_idx = self.base_model_names.index(best_base_name)
        best_base_oof = oof_predictions[:, best_base_idx]

        # Top-K非负加权融合
        blend_oof, blend_metrics = self.build_weighted_blend(
            oof_predictions,
            cv_scores,
            top_k=self.top_k_models,
        )
        self.blend_metrics = blend_metrics
        self.selected_base_models = blend_metrics['selected_models']
        self.blend_weights = np.array(blend_metrics['weights'], dtype=float)

        # 在小样本/极不平衡下，融合不占优时回退冠军单模型
        if float(blend_metrics['auc']) >= best_base_auc:
            self.final_strategy = 'weighted_blend'
            self.champion_model_name = None
            self.oof_final_pred = blend_oof
            print(
                f"\nFinal strategy: weighted_blend "
                f"(blend_auc={blend_metrics['auc']:.4f} >= best_base_auc={best_base_auc:.4f})"
            )
        else:
            self.final_strategy = 'best_base'
            self.champion_model_name = best_base_name
            self.oof_final_pred = best_base_oof
            print(
                f"\nFinal strategy: best_base='{best_base_name}' "
                f"(best_base_auc={best_base_auc:.4f} > blend_auc={blend_metrics['auc']:.4f})"
            )

        # 概率校准
        self.calibrator = self._fit_probability_calibrator(
            y_train,
            self.oof_final_pred,
            method=self.calibration_method,
        )
        if self.calibrator is not None:
            self.oof_final_pred_calibrated = self._apply_probability_calibration(self.oof_final_pred)
            pre_brier = float(brier_score_loss(np.asarray(y_train), np.asarray(self.oof_final_pred)))
            post_brier = float(brier_score_loss(np.asarray(y_train), np.asarray(self.oof_final_pred_calibrated)))
            print(
                f"Calibration: {self.calibration_method} "
                f"(Brier {pre_brier:.5f} -> {post_brier:.5f})"
            )
        else:
            self.oof_final_pred_calibrated = self.oof_final_pred

        # 基于最终策略的OOF预测选择阈值
        threshold_source = self.oof_final_pred_calibrated if self.oof_final_pred_calibrated is not None else self.oof_final_pred
        if threshold_source is not None:
            thr_rate = self._find_optimal_threshold(y_train, threshold_source, method='match_rate')
            thr_youden = self._find_optimal_threshold(y_train, threshold_source, method='youden')
            thr_cost = self._find_optimal_threshold(y_train, threshold_source, method='cost')

            if self.threshold_method == 'match_rate':
                self.decision_threshold = thr_rate
            elif self.threshold_method == 'youden':
                self.decision_threshold = thr_youden
            else:
                self.decision_threshold = thr_cost

            print(f"Selected decision threshold ({self.threshold_method})  : {self.decision_threshold:.6f}")
            print(f"Reference threshold (OOF match bad-rate)       : {thr_rate:.6f}")
            print(f"Reference threshold (OOF Youden/KS)            : {thr_youden:.6f}")
            print(f"Reference threshold (OOF cost-minimization)    : {thr_cost:.6f}")

        if threshold_source is not None:
            y_arr = np.asarray(y_train)
            y_prob = np.asarray(threshold_source)
            self.final_oof_metrics = {
                'auc': float(roc_auc_score(y_arr, y_prob)),
                'pr_auc': float(average_precision_score(y_arr, y_prob)),
                'ks': float(self._ks_stat(y_arr, y_prob)),
            }

        # 保存测试集特征
        self.X_test = X_test
        self.test_ids = self.test_df['id']

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

        final_auc = float(roc_auc_score(np.asarray(y_train), np.asarray(threshold_source)))
        return cv_scores, final_auc

    def _predict_weighted_blend(self, X_data):
        """
        使用加权融合进行预测

        Args:
            X_data (pd.DataFrame): 输入特征

        Returns:
            np.ndarray: 融合后的预测概率
        """
        if self.blend_weights is None or len(self.selected_base_models) == 0:
            raise RuntimeError('Weighted blend is not configured.')

        preds = []
        for name in self.selected_base_models:
            model = self.models[name]
            if not hasattr(model, 'predict_proba'):
                raise RuntimeError(f"Selected model '{name}' does not support predict_proba")
            preds.append(model.predict_proba(X_data)[:, 1])

        pred_matrix = np.column_stack(preds)
        return np.dot(pred_matrix, self.blend_weights)

    def predict(self):
        """
        生成预测结果

        根据最终策略（加权融合或最佳单模型）生成测试集预测。

        Returns:
            tuple: (calibrated_predictions, binary_predictions) 校准后的概率和二元预测
        """
        print("\n" + "=" * 80)
        print("Generating Predictions")
        print("=" * 80)

        # 若融合不如最佳单模型，则回退使用最佳单模型出分
        if getattr(self, 'final_strategy', 'weighted_blend') == 'best_base':
            name = getattr(self, 'champion_model_name', None)
            if name is None:
                raise RuntimeError("final_strategy is best_base but champion_model_name is None")

            print(f"Predicting with champion base model: {name}...")
            model = self.models[name]
            if not hasattr(model, 'predict_proba'):
                raise RuntimeError(f"Champion model '{name}' does not support predict_proba")
            final_predictions = model.predict_proba(self.X_test)[:, 1]
        else:
            print(f"Predicting with weighted blend: {self.selected_base_models}")
            final_predictions = self._predict_weighted_blend(self.X_test)

        # 概率校准
        calibrated_predictions = self._apply_probability_calibration(final_predictions)

        # 转换为二元分类
        threshold = getattr(self, 'decision_threshold', 0.5)
        binary_predictions = (calibrated_predictions >= threshold).astype(int)

        print(f"\nPrediction distribution:")
        print(f"  Mean probability (raw): {final_predictions.mean():.4f}")
        print(f"  Mean probability (cal): {calibrated_predictions.mean():.4f}")
        print(f"  Predicted defaults: {binary_predictions.sum()}")
        print(f"  Predicted default rate: {binary_predictions.mean():.4f}")
        print(f"  Decision threshold: {threshold:.6f}")

        return calibrated_predictions, binary_predictions

    def save_results(self, output_path='output/sub-test.csv'):
        """
        保存预测结果

        Args:
            output_path (str): 输出文件路径，默认为'output/sub-test.csv'

        Returns:
            pd.DataFrame: 预测结果数据框
        """
        final_prob, binary_predictions = self.predict()

        self.last_test_stats = {
            'n_test': int(len(binary_predictions)),
            'mean_prob': float(np.mean(final_prob)),
            'pred_pos': int(np.sum(binary_predictions)),
            'pred_pos_rate': float(np.mean(binary_predictions)),
        }

        # 创建结果DataFrame
        results = pd.DataFrame({
            'id': self.test_ids,
            'target': binary_predictions
        })

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存结果
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total predictions: {len(results)}")

        return results

    def build_threshold_decision_table(self, n_points=41):
        """
        基于OOF概率构建阈值-拦截率-成本决策表

        Args:
            n_points (int): 阈值采样点数，默认为41

        Returns:
            pd.DataFrame: 决策表数据框
        """
        if getattr(self, 'y_train', None) is None:
            raise RuntimeError('y_train is not available. Please train the model first.')

        oof_prob = self.oof_final_pred_calibrated if self.oof_final_pred_calibrated is not None else self.oof_final_pred
        if oof_prob is None:
            raise RuntimeError('OOF predictions are not available for decision table generation.')

        y_true = np.asarray(self.y_train)
        y_prob = np.asarray(oof_prob)

        n_points = int(max(3, n_points))
        thresholds = np.unique(np.concatenate([
            np.linspace(0.0, 1.0, n_points),
            np.array([self.decision_threshold], dtype=float),
        ]))

        rows = []
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            tp = int(np.sum((y_hat == 1) & (y_true == 1)))
            fp = int(np.sum((y_hat == 1) & (y_true == 0)))
            tn = int(np.sum((y_hat == 0) & (y_true == 0)))
            fn = int(np.sum((y_hat == 0) & (y_true == 1)))

            pred_pos = tp + fp
            reject_rate = float(pred_pos / len(y_true))
            approve_rate = float(1.0 - reject_rate)
            bad_recall = float(tp / max(tp + fn, 1))
            bad_precision = float(tp / max(tp + fp, 1))
            total_cost = float(self.cost_fp * fp + self.cost_fn * fn)

            rows.append({
                'threshold': float(thr),
                'reject_rate': reject_rate,
                'approve_rate': approve_rate,
                'bad_recall': bad_recall,
                'bad_precision': bad_precision,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'cost_fp': float(self.cost_fp),
                'cost_fn': float(self.cost_fn),
                'total_cost': total_cost,
            })

        df = pd.DataFrame(rows).sort_values(['total_cost', 'threshold'], ascending=[True, True]).reset_index(drop=True)
        return df

    def export_threshold_decision_table(self, path='output/threshold_decision_table.csv', n_points=41):
        """
        导出阈值决策表

        便于业务方选择拦截规模与成本权衡。

        Args:
            path (str): 输出文件路径，默认为'output/threshold_decision_table.csv'
            n_points (int): 阈值采样点数，默认为41

        Returns:
            pd.DataFrame: 决策表数据框
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        table = self.build_threshold_decision_table(n_points=n_points)
        table.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Threshold decision table saved to: {path}")
        return table

    def export_experiment_latex(self, path='docs/experiment_results.tex'):
        """
        导出LaTeX实验摘要

        导出一次运行的关键指标到LaTeX片段文件，供论文/报告直接\input引用。

        Args:
            path (str): 输出文件路径，默认为'docs/experiment_results.tex'
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        n_train = int(getattr(self, 'X_train', pd.DataFrame()).shape[0]) if getattr(self, 'X_train', None) is not None else None
        y_train = getattr(self, 'y_train', None)
        bad_rate = float(np.mean(y_train)) if y_train is not None else None

        lines = []
        lines.append('% Auto-generated by src/credit_risk_model.py. DO NOT EDIT MANUALLY.')
        lines.append(f'% Generated at: {now}')
        lines.append('')
        lines.append('\\subsection{自动化实验结果摘要（自动生成）}')

        if n_train is not None and bad_rate is not None:
            lines.append(f'训练集样本数：{n_train}，训练集坏样本率：{bad_rate:.4f}。')

        lines.append('')
        lines.append('\\paragraph{评估与出分策略} 本项目采用OOF评估，并支持在小样本下对融合策略进行稳健性回退。')
        lines.append('')

        # Metrics block
        strategy = getattr(self, 'final_strategy', 'weighted_blend')
        champion = getattr(self, 'champion_model_name', '') or ''
        thr = float(getattr(self, 'decision_threshold', 0.5))

        lines.append('\\begin{itemize}')
        if self.tabpfn_requested:
            if self.tabpfn_ready:
                lines.append('  \\item TabPFN状态：已启用（通过运行时可用性自检）')
            else:
                reason = self._latex_escape(self.tabpfn_unavailable_reason)
                lines.append(f'  \\item TabPFN状态：已请求但未启用（原因：{reason}）')
        else:
            lines.append('  \\item TabPFN状态：未请求启用')

        if strategy == 'best_base':
            lines.append(f'  \\item 最终策略：best\\_base（回退冠军单模型：{champion}）')
        else:
            lines.append('  \\item 最终策略：weighted\\_blend（Top-K非负加权融合）')

        if getattr(self, 'selected_base_models', None):
            pairs = []
            for n, w in zip(self.selected_base_models, np.asarray(self.blend_weights, dtype=float)):
                pairs.append(f'{self._latex_escape(n)}={w:.3f}')
            lines.append(f'  \\item 融合模型与权重：{"; ".join(pairs)}')

        if self.calibrator is not None:
            lines.append(f'  \\item 概率校准：{self._latex_escape(self.calibrator.get("method", "unknown"))}')

        lines.append(f'  \\item 阈值选择方法：{self._latex_escape(self.threshold_method)}（FP成本={self.cost_fp:.3f}, FN成本={self.cost_fn:.3f}）')
        lines.append(f'  \\item 决策阈值（OOF自动选择）：{thr:.6f}')

        m = getattr(self, 'final_oof_metrics', None)
        if m:
            lines.append(f'  \\item 最终策略OOF AUC：{m.get("auc", float("nan")):.4f}；OOF PR-AUC：{m.get("pr_auc", float("nan")):.4f}；OOF KS：{m.get("ks", float("nan")):.4f}')
        lines.append('\\end{itemize}')

        # Base model CV AUC table
        cv = getattr(self, 'cv_scores', None)
        if cv:
            lines.append('')
            lines.append('\\paragraph{基模型CV AUC（OOF泄露控制）}')
            lines.append('\\begin{table}[htbp]')
            lines.append('\\centering')
            lines.append('\\caption{基模型5折CV AUC汇总（自动生成）}')
            lines.append('\\begin{tabular}{lcc}')
            lines.append('\\toprule')
            lines.append('\\textbf{模型} & \\textbf{AUC均值} & \\textbf{95\\% CI} \\\\')
            lines.append('\\midrule')
            for name, sc in cv.items():
                mean = float(sc.get('mean', float('nan')))
                ci_low = float(sc.get('ci95_low', float('nan')))
                ci_high = float(sc.get('ci95_high', float('nan')))
                lines.append(f'{self._latex_escape(name)} & {mean:.4f} & [{ci_low:.4f}, {ci_high:.4f}] \\\\')
            lines.append('\\bottomrule')
            lines.append('\\end{tabular}')
            lines.append('\\end{table}')

        lines.append('')
        lines.append('% End of auto-generated content.')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"LaTeX experiment summary saved to: {path}")

    def get_feature_importance(self):
        """
        获取特征重要性

        计算各基础模型的特征重要性并返回平均值排序后的结果。

        Returns:
            list: 按重要性排序的(特征名, 重要性分数)列表
        """
        importance_dict = {}

        # XGBoost重要性
        if 'xgboost' in self.models:
            importance_dict['xgboost'] = dict(zip(
                self.feature_cols,
                self.models['xgboost'].feature_importances_
            ))

        # LightGBM重要性
        if 'lightgbm' in self.models:
            importance_dict['lightgbm'] = dict(zip(
                self.feature_cols,
                self.models['lightgbm'].feature_importances_
            ))

        # 计算平均重要性
        avg_importance = {}
        for feature in self.feature_cols:
            scores = []
            for model_name, imp_dict in importance_dict.items():
                scores.append(imp_dict.get(feature, 0))
            avg_importance[feature] = np.mean(scores)

        # 排序
        sorted_importance = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_importance

    def save_model(self, path='models/credit_risk_model.pkl'):
        """
        保存模型

        将训练好的模型保存到指定路径。

        Args:
            path (str): 保存路径，默认为'models/credit_risk_model.pkl'
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to: {path}")


def main():
    """
    主函数

    解析命令行参数并执行模型训练和预测流程。
    """
    print("\n" + "=" * 80)
    print("全球顶级金融风控策略 - 信用违约预测系统")
    print("Global Top-Tier Financial Risk Control Strategy")
    print("=" * 80)

    parser = argparse.ArgumentParser(description='Credit risk model training and inference')
    parser.add_argument('--train-path', type=str, default='data/训练数据集.csv')
    parser.add_argument('--test-path', type=str, default='data/测试集.csv')
    parser.add_argument('--output-path', type=str, default='output/sub-test.csv')
    parser.add_argument('--use-smote', action='store_true', help='Enable fold-wise SMOTE during training')
    parser.add_argument('--disable-tabpfn', action='store_true', help='Disable TabPFN base model')
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--cv-repeats', type=int, default=20)
    parser.add_argument('--top-k-models', type=int, default=3)
    parser.add_argument('--threshold-method', type=str, default='cost', choices=['cost', 'match_rate', 'youden'])
    parser.add_argument('--cost-fp', type=float, default=0.05)
    parser.add_argument('--cost-fn', type=float, default=1.0)
    parser.add_argument('--decision-table-path', type=str, default='output/threshold_decision_table.csv')
    parser.add_argument('--decision-table-points', type=int, default=41)

    args = parser.parse_args()

    # 初始化模型
    model = CreditRiskModel()
    model.cv_splits = int(max(2, args.cv_splits))
    model.cv_repeats = int(max(1, args.cv_repeats))
    model.top_k_models = int(max(1, args.top_k_models))
    model.threshold_method = str(args.threshold_method)
    model.cost_fp = float(max(0.0, args.cost_fp))
    model.cost_fn = float(max(0.0, args.cost_fn))

    # 定义路径
    train_path = args.train_path
    test_path = args.test_path
    output_path = args.output_path

    # 训练模型
    cv_scores, final_auc = model.train(
        train_path,
        test_path,
        use_smote=bool(args.use_smote),
        use_tabpfn=(not bool(args.disable_tabpfn)),
    )

    # 打印模型性能总结
    print("\n" + "=" * 80)
    print("Model Performance Summary")
    print("=" * 80)

    for name, scores in cv_scores.items():
        print(
            f"{name:20s}: AUC = {scores['mean']:.4f} "
            f"(95% CI: {scores['ci95_low']:.4f}, {scores['ci95_high']:.4f})"
        )

    print(f"{'Final Strategy AUC':20s}: AUC = {final_auc:.4f}")

    # 元模型OOF指标
    if getattr(model, 'blend_metrics', None):
        print("\n" + "=" * 80)
        print("Final Strategy OOF Metrics")
        print("=" * 80)
        m = model.blend_metrics
        print(f"OOF PR-AUC            : {m['pr_auc']:.4f}")
        print(f"OOF KS                : {m['ks']:.4f}")
        print(f"Selected threshold    : {getattr(model, 'decision_threshold', 0.5):.6f}")
        print(f"Threshold method      : {getattr(model, 'threshold_method', 'cost')}")
        if getattr(model, 'calibrator', None) is not None:
            print(f"Calibration method    : {model.calibrator.get('method', 'unknown')}")
        if getattr(model, 'selected_base_models', None):
            print(f"Selected models       : {model.selected_base_models}")
            print(f"Blend weights         : {np.round(model.blend_weights, 4).tolist()}")

        print(f"Final strategy        : {getattr(model, 'final_strategy', 'weighted_blend')}")
        if getattr(model, 'final_strategy', 'weighted_blend') == 'best_base':
            print(f"Champion base model   : {getattr(model, 'champion_model_name', '')}")

        if getattr(model, 'oof_final_pred', None) is not None and getattr(model, 'y_train', None) is not None:
            y_true = np.asarray(model.y_train)
            y_prob = np.asarray(model.oof_final_pred)
            thr = float(getattr(model, 'decision_threshold', 0.5))
            y_hat = (y_prob >= thr).astype(int)
            print("\nConfusion Matrix (OOF @ threshold):")
            print(confusion_matrix(y_true, y_hat))

    # 特征重要性
    print("\n" + "=" * 80)
    print("Top 15 Feature Importance")
    print("=" * 80)

    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(importance[:15]):
        print(f"{i+1:2d}. {feature:30s}: {score:.4f}")

    # 生成并保存预测结果
    results = model.save_results(output_path)

    # 导出阈值决策表
    model.export_threshold_decision_table(
        path=args.decision_table_path,
        n_points=args.decision_table_points,
    )

    # 导出LaTeX实验摘要
    model.export_experiment_latex('docs/experiment_results.tex')

    # 保存模型
    model.save_model('models/credit_risk_model.pkl')

    print("\n" + "=" * 80)
    print("All tasks completed successfully!")
    print("=" * 80)

    return model, results


if __name__ == '__main__':
    model, results = main()
