"""
V9.0 LightGBM大文件数据加载器
专门处理大型JSON文件的高效加载
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import gc

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("⚠️ ijson未安装，将使用标准JSON解析（内存使用较高）")


def iter_large_json_file(json_path: Path, max_samples: Optional[int] = None) -> Iterable[Dict]:
    """
    使用流式解析迭代大型JSON文件
    
    Args:
        json_path: JSON文件路径
        max_samples: 最大样本数限制
        
    Yields:
        JSON样本字典
    """
    if not HAS_IJSON:
        # 如果没有ijson，回退到标准方法但限制样本数
        print("⚠️ 使用标准JSON解析，建议安装ijson以获得更好性能")
        try:
            data = json.loads(json_path.read_text())
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break
                yield item
        except Exception as e:
            print(f"标准解析失败: {e}")
            return
        return
    
    count = 0
    
    try:
        # 使用ijson进行流式解析，避免一次性加载整个文件到内存
        with open(json_path, 'rb') as file:
            parser = ijson.parse(file)
            current_sample = {}
            current_key = None
            in_array = False
            depth = 0
            
            for prefix, event, value in parser:
                if event == 'start_array':
                    if prefix == '':
                        in_array = True
                    depth += 1
                elif event == 'end_array':
                    depth -= 1
                    if depth == 0 and current_sample:
                        yield current_sample
                        current_sample = {}
                        count += 1
                        
                        if max_samples and count >= max_samples:
                            break
                            
                        # 定期清理内存
                        if count % 1000 == 0:
                            gc.collect()
                            
                elif event == 'start_map':
                    if depth == 1:  # 新的样本开始
                        current_sample = {}
                elif event == 'map_key':
                    current_key = value
                elif event in ('string', 'number', 'boolean', 'null'):
                    if current_key and depth >= 1:
                        # 处理嵌套结构
                        if '.' in prefix:
                            parts = prefix.split('.')
                            obj = current_sample
                            for part in parts[1:-1]:  # 跳过数组索引
                                if part.isdigit():
                                    continue
                                if part not in obj:
                                    obj[part] = {}
                                obj = obj[part]
                            obj[current_key] = value
                        else:
                            current_sample[current_key] = value
                            
    except Exception as e:
        print(f"流式解析失败，回退到标准解析: {e}")
        # 回退到标准JSON解析（但有内存限制）
        try:
            data = json.loads(json_path.read_text())
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break
                yield item
                count += 1
        except Exception as fallback_error:
            print(f"标准解析也失败: {fallback_error}")
            return


def iter_json_samples_efficient(json_path: Path, max_samples: Optional[int] = None) -> Iterable[Dict]:
    """
    高效的JSON样本迭代器 - 自动选择最佳加载策略
    
    Args:
        json_path: JSON文件路径  
        max_samples: 最大样本数限制
        
    Yields:
        JSON样本字典
    """
    file_size_mb = json_path.stat().st_size / (1024 * 1024)
    
    print(f"📁 文件大小: {file_size_mb:.1f} MB")
    
    if file_size_mb > 10000:  # 大于10GB使用流式解析 (临时修复)
        print("🔄 使用流式解析处理大文件...")
        yield from iter_large_json_file(json_path, max_samples)
    else:
        print("🔄 使用标准解析处理小文件...")
        # 小文件直接加载
        try:
            data = json.loads(json_path.read_text())
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break
                yield item
        except Exception as e:
            print(f"标准解析失败: {e}")
            return


def count_samples_in_file(json_path: Path) -> int:
    """
    快速计算JSON文件中的样本数量（不加载到内存）
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        样本数量
    """
    count = 0
    
    try:
        with open(json_path, 'rb') as file:
            parser = ijson.parse(file)
            depth = 0
            
            for prefix, event, value in parser:
                if event == 'start_array':
                    depth += 1
                elif event == 'end_array':
                    depth -= 1
                elif event == 'start_map' and depth == 1:
                    count += 1
                    
                # 定期输出进度
                if count % 10000 == 0 and count > 0:
                    print(f"   已统计 {count:,} 个样本...")
                    
    except Exception as e:
        print(f"快速统计失败: {e}")
        return -1
    
    return count


def get_sample_preview(json_path: Path, num_samples: int = 3) -> List[Dict]:
    """
    获取文件前几个样本的预览
    
    Args:
        json_path: JSON文件路径
        num_samples: 预览样本数
        
    Returns:
        样本列表
    """
    samples = []
    for i, sample in enumerate(iter_json_samples_efficient(json_path, max_samples=num_samples)):
        samples.append(sample)
        if i >= num_samples - 1:
            break
    return samples