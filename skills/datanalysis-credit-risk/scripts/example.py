#!/usr/bin/env python3
import os, sys
import time
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import multiprocessing

# =============================================================================
# 系统配置
# =============================================================================
CPU_COUNT = multiprocessing.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1)  # 多进程并行数，保留1核给系统

def _ensure_references_on_path():
    script_dir = os.path.dirname(__file__)
    cur = script_dir
    for _ in range(8):
        candidate = os.path.join(cur, 'references')
        if os.path.isdir(candidate):
            # add parent folder (which contains `references`) to sys.path
            sys.path.insert(0, cur)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: add a reasonable repo-root guess
    sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', '..', '..')))


_ensure_references_on_path()

from references.func import get_dataset, missing_check, org_analysis
from references.analysis import (drop_abnormal_ym, drop_highmiss_features,
                               drop_lowiv_features, drop_highcorr_features,
                               drop_highpsi_features,
                               drop_highnoise_features,
                               export_cleaning_report,
                               iv_distribution_by_org,
                               psi_distribution_by_org,
                               value_ratio_distribution_by_org)

# ==================== 路径配置（可交互输入） ====================
# 使用50列的测试数据作为默认值，支持在命令行交互修改
default_data_path = r'c:\Users\jp341\Desktop\ai_2026\AutoModeling\data\sample_50cols.parquet'
default_output_dir = r'c:\Users\jp341\Desktop\ai_2026\AutoModeling\output'

def _get_path_input(prompt, default):
    try:
        user_val = input(f"{prompt} (默认: {default}): ").strip()
    except Exception:
        user_val = ''
    return user_val if user_val else default

DATA_PATH = _get_path_input('请输入数据文件路径DATA_PATH', default_data_path)
OUTPUT_DIR = _get_path_input('请输入输出目录OUTPUT_DIR', default_output_dir)
REPORT_PATH = os.path.join(OUTPUT_DIR, '数据清洗报告.xlsx')

# 数据列名配置（根据实际数据调整）
DATE_COL = _get_path_input('请输入数据中日期列名', 'apply_date')
Y_COL = _get_path_input('请输入数据中标签列名', 'target')
ORG_COL = _get_path_input('请输入数据中机构列名', 'org_info')

# 支持多个主键列名输入（逗号或空格分隔）
def _get_list_input(prompt, default):
    try:
        user_val = input(f"{prompt} (默认: {default}): ").strip()
    except Exception:
        user_val = ''
    if not user_val:
        user_val = default
    # 支持逗号或空格分隔
    parts = [p.strip() for p in user_val.replace(',', ' ').split() if p.strip()]
    return parts

KEY_COLS = _get_list_input('请输入数据中主键列名（多个列用逗号或空格分隔）', 'record_id')

# ==================== 多进程配置信息 ====================
print("=" * 60)
print("多进程配置")
print("=" * 60)
print(f"   本机CPU核心数: {CPU_COUNT}")
print(f"   当前使用进程数: {N_JOBS}")
print("=" * 60)

# ==================== OOS机构配置（可交互输入） ====================
# 默认贷外机构列表，用户可在交互时以逗号分隔形式输入自定义列表
default_oos = [
    '外部机构A', '外部机构B', '外部机构C'
]

try:
    oos_input = input('请输入贷外机构列表，逗号分隔（按回车使用默认列表）：').strip()
except Exception:
    oos_input = ''
if oos_input:
    OOS_ORGS = [s.strip() for s in oos_input.split(',') if s.strip()]
else:
    OOS_ORGS = default_oos

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 交互式输入超参数 ====================
def get_user_input(prompt, default, dtype=float):
    """获取用户输入，支持默认值和类型转换"""
    while True:
        try:
            user_input = input(f"{prompt} (默认: {default}): ").strip()
            if not user_input:
                return default
            return dtype(user_input)
        except ValueError:
            print(f"   输入无效，请输入{dtype.__name__}类型")

# 记录清洗步骤
steps = []

# 用于存储各步骤的参数
params = {}

# 计时装饰器
def timer(step_name):
    """计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n开始 {step_name}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"   {step_name} 耗时: {elapsed:.2f}秒")
            return result
        return wrapper
    return decorator

# ==================== Step 1: 获取数据 ====================
print("\n" + "=" * 60)
print("Step 1: 获取数据")
print("=" * 60)
step_start = time.time()
# 使用global_parameters中的配置
data = get_dataset(
    data_pth=DATA_PATH,
    date_colName=DATE_COL,
    y_colName=Y_COL,
    org_colName=ORG_COL,
    data_encode='utf-8',
    key_colNames=KEY_COLS,
    drop_colNames=[],
    miss_vals=[-1, -999, -1111]
)
print(f"   原始数据: {data.shape}")
print(f"   异常值已替换为NaN: [-1, -999, -1111]")
print(f"   Step 1 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 2: 机构样本分析 ====================
print("\n" + "=" * 60)
print("Step 2: 机构样本分析")
print("=" * 60)
step_start = time.time()
org_stat = org_analysis(data, oos_orgs=OOS_ORGS)
steps.append(('机构样本统计', org_stat))
print(f"   机构数: {data['new_org'].nunique()}, 月份数: {data['new_date_ym'].nunique()}")
print(f"   贷外机构: {len(OOS_ORGS)}个")
print(f"   Step 2 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 3: 分离OOS数据 ====================
print("\n" + "=" * 60)
print("Step 3: 分离OOS数据")
print("=" * 60)
step_start = time.time()
oos_data = data[data['new_org'].isin(OOS_ORGS)]
data = data[~data['new_org'].isin(OOS_ORGS)]
print(f"   OOS样本: {oos_data.shape[0]}条")
print(f"   建模样本: {data.shape[0]}条")
print(f"   OOS机构: {OOS_ORGS}")
print(f"   Step 3 耗时: {time.time() - step_start:.2f}秒")
# 创建分离信息DataFrame
oos_info = pd.DataFrame({'变量': ['OOS样本', '建模样本'], '数量': [oos_data.shape[0], data.shape[0]]})
steps.append(('分离OOS数据', oos_info))

# ==================== Step 4: 过滤异常月份（仅对建模数据） ====================
print("\n" + "=" * 60)
print("Step 4: 过滤异常月份（仅对建模数据）")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['min_ym_bad_sample'] = int(get_user_input("坏样本数阈值", 10, int))
params['min_ym_sample'] = int(get_user_input("总样本数阈值", 500, int))
step_start = time.time()
data_filtered, abnormal_ym = drop_abnormal_ym(data.copy(), min_ym_bad_sample=params['min_ym_bad_sample'], min_ym_sample=params['min_ym_sample'])
steps.append(('Step4-异常月份处理', abnormal_ym))
print(f"   过滤后: {data_filtered.shape}")
print(f"   参数: min_ym_bad_sample={params['min_ym_bad_sample']}, min_ym_sample={params['min_ym_sample']}")
if len(abnormal_ym) > 0:
    print(f"   剔除月份: {abnormal_ym['年月'].tolist()}")
    print(f"   去除条件: {abnormal_ym['去除条件'].tolist()}")
print(f"   Step 4 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 5: 计算缺失率 ====================
print("\n" + "=" * 60)
print("Step 5: 计算缺失率")
print("=" * 60)
step_start = time.time()
orgs = data['new_org'].unique().tolist()
channel = {'整体': orgs}
miss_detail, miss_channel = missing_check(data, channel=channel)
# miss_detail: 缺失率明细（格式：变量，整体，机构1，机构2，...，机构n）
# miss_channel: 整体缺失率
steps.append(('缺失率明细', miss_detail))
print(f"   变量数: {len(miss_detail['变量'].unique())}")
print(f"   机构数: {len(miss_detail.columns) - 2}")  # 减去'变量'和'整体'两列
print(f"   Step 5 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 6: 剔除高缺失变量 ====================
print("\n" + "=" * 60)
print("Step 6: 剔除高缺失变量")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['missing_ratio'] = get_user_input("缺失率阈值", 0.6)
step_start = time.time()
data_miss, dropped_miss = drop_highmiss_features(data.copy(), miss_channel, threshold=params['missing_ratio'])
steps.append(('Step6-高缺失率处理', dropped_miss))
print(f"   剔除: {len(dropped_miss)}个")
print(f"   阈值: {params['missing_ratio']}")
if len(dropped_miss) > 0:
    print(f"   剔除变量: {dropped_miss['变量'].tolist()[:5]}...")
    print(f"   去除条件: {dropped_miss['去除条件'].tolist()[:5]}...")
print(f"   Step 6 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 7: 剔除低IV变量 ====================
print("\n" + "=" * 60)
print("Step 7: 剔除低IV变量")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['overall_iv_threshold'] = get_user_input("整体IV阈值", 0.1)
params['org_iv_threshold'] = get_user_input("单机构IV阈值", 0.1)
params['max_org_threshold'] = int(get_user_input("最大容忍低IV机构数", 2, int))
step_start = time.time()
# 获取特征列表（使用全部变量）
features = [c for c in data.columns if c.startswith('i_')]
data_iv, iv_detail, iv_process = drop_lowiv_features(
    data.copy(), features,
    overall_iv_threshold=params['overall_iv_threshold'],
    org_iv_threshold=params['org_iv_threshold'],
    max_org_threshold=params['max_org_threshold'],
    n_jobs=N_JOBS
)
# iv_detail: IV明细（每个变量在每个机构上以及整体上的IV值）
# iv_process: IV处理表（不满足设定条件的变量）
steps.append(('Step7-IV处理', iv_process))
print(f"   剔除: {len(iv_process)}个")
print(f"   参数: overall_iv_threshold={params['overall_iv_threshold']}, org_iv_threshold={params['org_iv_threshold']}, max_org_threshold={params['max_org_threshold']}")
if len(iv_process) > 0:
    print(f"   剔除变量: {iv_process['变量'].tolist()[:5]}...")
    print(f"   处理原因: {iv_process['处理原因'].tolist()[:5]}...")
print(f"   Step 7 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 8: 剔除高PSI变量 ====================
print("\n" + "=" * 60)
print("Step 8: 剔除高PSI变量 (分机构+逐月份)")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['psi_threshold'] = get_user_input("PSI阈值", 0.1)
params['max_months_ratio'] = get_user_input("最大不稳定月份比例", 1/3)
params['max_orgs'] = int(get_user_input("最大不稳定机构数", 6, int))
step_start = time.time()
# 获取PSI计算前的特征（使用全部变量）
features_for_psi = [c for c in data.columns if c.startswith('i_')]
data_psi, psi_detail, psi_process = drop_highpsi_features(
    data.copy(), features_for_psi,
    psi_threshold=params['psi_threshold'],
    max_months_ratio=params['max_months_ratio'],
    max_orgs=params['max_orgs'],
    min_sample_per_month=100,
    n_jobs=N_JOBS
)
# psi_detail: PSI明细（每个变量在每个机构每个月上的PSI值）
# psi_process: PSI处理表（不满足设定条件的变量）
steps.append(('Step8-PSI处理', psi_process))
print(f"   剔除: {len(psi_process)}个")
print(f"   参数: psi_threshold={params['psi_threshold']}, max_months_ratio={params['max_months_ratio']:.2f}, max_orgs={params['max_orgs']}")
if len(psi_process) > 0:
    print(f"   剔除变量: {psi_process['变量'].tolist()[:5]}...")
    print(f"   处理原因: {psi_process['处理原因'].tolist()[:5]}...")
print(f"   PSI明细: {len(psi_detail)}条")
print(f"   Step 8 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 9: Null Importance去噪 ====================
print("\n" + "=" * 60)
print("Step 9: Null Importance去除高噪音变量")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['n_estimators'] = int(get_user_input("树的数量", 100, int))
params['max_depth'] = int(get_user_input("树的最大深度", 5, int))
params['gain_threshold'] = get_user_input("gain差值阈值", 50)
step_start = time.time()
# 获取特征列表（使用全部变量）
features = [c for c in data.columns if c.startswith('i_')]
data_noise, dropped_noise = drop_highnoise_features(data.copy(), features, n_estimators=params['n_estimators'], max_depth=params['max_depth'], gain_threshold=params['gain_threshold'])
steps.append(('Step9-null importance处理', dropped_noise))
print(f"   剔除: {len(dropped_noise)}个")
print(f"   参数: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, gain_threshold={params['gain_threshold']}")
if len(dropped_noise) > 0:
    print(f"   剔除变量: {dropped_noise['变量'].tolist()}")
print(f"   Step 9 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 10: 剔除高相关变量（基于Null Importance的原始gain） ====================
print("\n" + "=" * 60)
print("Step 10: 剔除高相关变量（基于Null Importance的原始gain）")
print("=" * 60)
print("   可直接按回车使用默认值")
print("=" * 60)
params['max_corr'] = get_user_input("相关性阈值", 0.9)
params['top_n_keep'] = int(get_user_input("保留原始gain排名前N的变量", 20, int))
step_start = time.time()
# 获取特征列表（使用全部变量）
features = [c for c in data.columns if c.startswith('i_')]
# 从null importance结果中获取原始gain
if len(dropped_noise) > 0 and '原始gain' in dropped_noise.columns:
    gain_dict = dict(zip(dropped_noise['变量'], dropped_noise['原始gain']))
else:
    gain_dict = {}
data_corr, dropped_corr = drop_highcorr_features(data.copy(), features, threshold=params['max_corr'], gain_dict=gain_dict, top_n_keep=params['top_n_keep'])
steps.append(('Step10-高相关性剔除', dropped_corr))
print(f"   剔除: {len(dropped_corr)}个")
print(f"   阈值: {params['max_corr']}")
if len(dropped_corr) > 0:
    print(f"   剔除变量: {dropped_corr['变量'].tolist()}")
    print(f"   去除条件: {dropped_corr['去除条件'].tolist()[:5]}...")
print(f"   Step 10 耗时: {time.time() - step_start:.2f}秒")

# ==================== Step 11: 导出报告 ====================
print("\n" + "=" * 60)
print("Step 11: 导出报告")
print("=" * 60)
step_start = time.time()

# 计算IV分布统计
print("   计算IV分布统计...")
iv_distribution = iv_distribution_by_org(iv_detail, oos_orgs=OOS_ORGS)
print(f"   IV分布统计: {len(iv_distribution)}条")

# 计算PSI分布统计
print("   计算PSI分布统计...")
psi_distribution = psi_distribution_by_org(psi_detail, oos_orgs=OOS_ORGS)
print(f"   PSI分布统计: {len(psi_distribution)}条")

# 计算有值率分布统计（使用全部变量）
print("   计算有值率分布统计...")
features_for_value_ratio = [c for c in data.columns if c.startswith('i_')]
value_ratio_distribution = value_ratio_distribution_by_org(data, features_for_value_ratio, oos_orgs=OOS_ORGS)
print(f"   有值率分布统计: {len(value_ratio_distribution)}条")

# 添加明细和分布统计到steps列表
steps.append(('Step7-IV明细', iv_detail))
steps.append(('Step7-IV分布统计', iv_distribution))
steps.append(('Step8-PSI明细', psi_detail))
steps.append(('Step8-PSI分布统计', psi_distribution))
steps.append(('Step5-有值率分布统计', value_ratio_distribution))

export_cleaning_report(REPORT_PATH, steps,
                      iv_detail=iv_detail,
                      iv_process=iv_process,
                      psi_detail=psi_detail,
                      psi_process=psi_process,
                      params=params,
                      iv_distribution=iv_distribution,
                      psi_distribution=psi_distribution,
                      value_ratio_distribution=value_ratio_distribution)
print(f"   报告: {REPORT_PATH}")
print(f"   Step 11 耗时: {time.time() - step_start:.2f}秒")

# ==================== 汇总 ====================
print("\n" + "=" * 60)
print("数据清洗完成!")
print("=" * 60)
print(f"   原始数据: {data.shape[0]}条")
print(f"   原始变量: {len([c for c in data.columns if c.startswith('i_')])}个")
print(f"   清洗步骤（各步骤独立执行，不删除数据）:")
for name, df in steps:
    print(f"     - {name}: 剔除{df.shape[0] if hasattr(df, 'shape') else len(df)}个")
