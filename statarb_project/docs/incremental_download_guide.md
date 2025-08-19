# 增量下载功能指南 / Incremental Download Guide

## 概述 / Overview

The Binance data downloader now supports incremental download functionality, allowing you to efficiently update existing datasets without re-downloading overlapping data.

## 核心功能 / Core Features

### 1. 智能时间范围检测 / Smart Time Range Detection
- 自动读取已存在文件的开始和结束时间 / Automatically reads start/end times from existing files
- 支持新的文件命名格式 `{symbol}_{interval}.csv` / Supports new filename format
- 精确到分钟级别的时间范围检测 / Minute-level precision for time range detection

### 2. 增量下载策略 / Incremental Download Strategy

系统会根据请求时间范围和现有文件时间范围的关系，智能确定需要下载的数据段：

The system intelligently determines which data segments to download based on the relationship between requested and existing time ranges:

#### 情况1：向前扩展 / Case 1: Backward Extension
- **条件**: 请求开始时间 < 文件开始时间 / Requested start < file start
- **操作**: 下载从请求开始时间到文件开始时间的数据 / Download from requested start to file start
- **示例**: 文件覆盖2023-01-15到2023-01-31，请求2023-01-01到2023-01-20，则下载2023-01-01到2023-01-15

#### 情况2：向后扩展 / Case 2: Forward Extension  
- **条件**: 请求结束时间 > 文件结束时间 / Requested end > file end
- **操作**: 下载从文件结束时间到请求结束时间的数据 / Download from file end to requested end
- **示例**: 文件覆盖2023-01-01到2023-01-15，请求2023-01-10到2023-01-31，则下载2023-01-15到2023-01-31

#### 情况3：填补间隙 / Case 3: Gap Filling
- **前置间隙**: 请求结束时间 < 文件开始时间，下载连接数据 / Pre-gap: requested end < file start, download connecting data
- **后置间隙**: 请求开始时间 > 文件结束时间，下载连接数据 / Post-gap: requested start > file end, download connecting data

#### 情况4：无需下载 / Case 4: No Download Needed
- **条件**: 请求时间范围完全被现有文件覆盖 / Requested range fully covered by existing file
- **操作**: 跳过下载，返回现有文件记录数 / Skip download, return existing record count

### 3. 数据合并策略 / Data Merging Strategy

- **自动去重**: 基于 `open_time` 字段去除重复记录 / Automatic deduplication based on `open_time` field
- **时间排序**: 合并后按时间顺序排列所有记录 / Time sorting: All records sorted chronologically after merge
- **完整性保证**: 确保最终文件包含所有时间段的连续数据 / Integrity guarantee: Ensure final file contains continuous data for all time periods

## 使用方法 / Usage

### 命令行参数 / Command Line Arguments

```bash
# 启用增量下载（默认）/ Enable incremental download (default)
python binance_data_downloader.py --symbols BTCUSDT --incremental

# 禁用增量下载，总是下载完整范围 / Disable incremental, always download full range
python binance_data_downloader.py --symbols BTCUSDT --no-incremental

# 强制重新下载，忽略现有文件 / Force redownload, ignore existing files
python binance_data_downloader.py --symbols BTCUSDT --force-redownload
```

### 实际示例 / Practical Examples

#### 示例1：首次下载 / Example 1: Initial Download
```bash
python binance_data_downloader.py --symbols BTCUSDT --start-time 2023-01-01 --end-time 2023-01-31
```
- 创建文件：`data/futures_5m/BTCUSDT_5m.csv`
- 包含2023年1月的所有数据

#### 示例2：扩展到更早时间 / Example 2: Extend to Earlier Time
```bash
python binance_data_downloader.py --symbols BTCUSDT --start-time 2022-12-01 --end-time 2023-01-15
```
- 检测到现有文件从2023-01-01开始
- 只下载2022-12-01到2023-01-01的数据
- 合并到现有文件中

#### 示例3：扩展到更晚时间 / Example 3: Extend to Later Time
```bash
python binance_data_downloader.py --symbols BTCUSDT --start-time 2023-01-15 --end-time 2023-02-28
```
- 检测到现有文件到2023-01-31结束
- 只下载2023-01-31到2023-02-28的数据
- 合并到现有文件中

#### 示例4：填补时间间隙 / Example 4: Fill Time Gap
```bash
python binance_data_downloader.py --symbols BTCUSDT --start-time 2023-03-01 --end-time 2023-03-31
```
- 检测到现有文件结束于2023-01-31
- 下载2023-01-31到2023-03-31的连接数据
- 填补2月份的间隙并包含3月份数据

## 技术实现 / Technical Implementation

### 关键函数 / Key Functions

1. **`get_existing_file_time_range()`**: 读取现有文件的时间范围
2. **`calculate_download_ranges()`**: 计算需要下载的时间段
3. **`merge_temp_files_with_existing()`**: 合并新下载数据与现有数据
4. **`download_symbol_data()`**: 主下载函数，支持增量逻辑

### 文件格式 / File Format

新的文件命名格式：`{symbol}_{interval}.csv`
- 示例：`BTCUSDT_5m.csv`、`ETHUSDT_1h.csv`
- 支持所有标准时间间隔：1m, 5m, 15m, 30m, 1h, 4h, 1d等

### 性能优化 / Performance Optimization

- **智能跳过**: 无需下载时直接返回，避免不必要的API调用
- **分块下载**: 大时间范围自动分块，支持断点续传
- **并发控制**: 维持原有的并发下载能力
- **内存优化**: 使用pandas进行高效的数据操作和去重

## 最佳实践 / Best Practices

1. **定期更新**: 使用增量模式定期更新数据，避免重复下载
2. **验证数据**: 检查合并后的数据连续性和完整性
3. **备份重要数据**: 在大规模更新前备份现有文件
4. **监控日志**: 关注下载日志中的时间范围信息

## 错误处理 / Error Handling

- **文件读取失败**: 自动回退到完整下载模式
- **时间解析错误**: 记录警告并使用默认行为
- **数据合并错误**: 保留原始文件，记录详细错误信息
- **网络中断**: 支持从中断点继续下载

## 兼容性 / Compatibility

- 向后兼容旧的文件格式
- 自动处理文件格式转换
- 支持所有现有的命令行参数
- 与现有的下载逻辑完全集成

---

通过这些增量下载功能，您可以更高效地维护和更新大规模的历史数据集，大大减少不必要的数据传输和存储空间占用。

With these incremental download features, you can efficiently maintain and update large-scale historical datasets, significantly reducing unnecessary data transfer and storage space usage.