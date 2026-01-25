"""
Token Skip 数据流追踪模块
用于记录所有进入 CUDA 的关键张量维度和数值

使用方法:
    from trace_dataflow import Tracer
    tracer = Tracer(enabled=True)
    # 在代码中调用
    tracer.log("event_name", tensor=x, extra_info="...")
    # 最后获取所有日志
    df = tracer.to_dataframe()
"""

import torch
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TraceEntry:
    """单条追踪记录"""
    timestamp: str
    event: str
    step: int = -1
    layer: int = -1
    block_idx: int = -1
    shapes: Dict[str, List[int]] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    extra: str = ""


class Tracer:
    """全局追踪器"""
    
    def __init__(self, enabled: bool = True, max_entries: int = 50000):
        self.enabled = enabled
        self.max_entries = max_entries
        self.entries: List[TraceEntry] = []
        self.current_step = -1
        self.current_layer = -1
        self.current_block_idx = -1
    
    def set_context(self, step: int = None, layer: int = None, block_idx: int = None):
        """设置当前上下文"""
        if step is not None:
            self.current_step = step
        if layer is not None:
            self.current_layer = layer
        if block_idx is not None:
            self.current_block_idx = block_idx
    
    def log(self, event: str, **kwargs):
        """
        记录一条追踪信息
        
        Args:
            event: 事件名称
            **kwargs: 可以是 tensor (自动提取 shape) 或其他值
        """
        if not self.enabled:
            return
        
        if len(self.entries) >= self.max_entries:
            return  # 防止内存溢出
        
        shapes = {}
        values = {}
        extra = ""
        
        for k, v in kwargs.items():
            if k == "extra":
                extra = str(v)
            elif isinstance(v, torch.Tensor):
                shapes[k] = list(v.shape)
                # 小张量可以记录实际值
                if v.numel() <= 10:
                    try:
                        values[k] = v.detach().cpu().tolist()
                    except:
                        pass
                # 记录一些统计信息
                if v.numel() > 0:
                    try:
                        values[f"{k}_dtype"] = str(v.dtype)
                        if v.is_floating_point():
                            values[f"{k}_min"] = float(v.min().item())
                            values[f"{k}_max"] = float(v.max().item())
                            values[f"{k}_mean"] = float(v.mean().item())
                    except:
                        pass
            elif isinstance(v, (int, float, str, bool)):
                values[k] = v
            elif isinstance(v, (list, tuple)) and len(v) <= 20:
                values[k] = list(v)
            else:
                values[k] = str(v)[:100]
        
        entry = TraceEntry(
            timestamp=datetime.now().isoformat(),
            event=event,
            step=self.current_step,
            layer=self.current_layer,
            block_idx=self.current_block_idx,
            shapes=shapes,
            values=values,
            extra=extra
        )
        self.entries.append(entry)
    
    def clear(self):
        """清空所有记录"""
        self.entries = []
        self.current_step = -1
        self.current_layer = -1
        self.current_block_idx = -1
    
    def to_list(self) -> List[Dict]:
        """转换为字典列表"""
        return [asdict(e) for e in self.entries]
    
    def to_dataframe(self):
        """转换为 pandas DataFrame"""
        import pandas as pd
        records = []
        for e in self.entries:
            record = {
                "timestamp": e.timestamp,
                "event": e.event,
                "step": e.step,
                "layer": e.layer,
                "block_idx": e.block_idx,
                "extra": e.extra
            }
            # 展平 shapes
            for k, v in e.shapes.items():
                record[f"shape_{k}"] = str(v)
            # 展平 values
            for k, v in e.values.items():
                record[f"val_{k}"] = str(v) if isinstance(v, (list, dict)) else v
            records.append(record)
        return pd.DataFrame(records)
    
    def save_json(self, path: str):
        """保存为 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2, ensure_ascii=False)
    
    def filter_events(self, event_pattern: str) -> List[TraceEntry]:
        """按事件名称过滤"""
        return [e for e in self.entries if event_pattern in e.event]
    
    def summary(self) -> str:
        """生成摘要"""
        from collections import Counter
        event_counts = Counter(e.event for e in self.entries)
        lines = [f"Total entries: {len(self.entries)}", "Events:"]
        for event, count in event_counts.most_common():
            lines.append(f"  {event}: {count}")
        return "\n".join(lines)
    
    def print_last(self, n: int = 10):
        """打印最后 n 条记录"""
        for e in self.entries[-n:]:
            print(f"[{e.event}] step={e.step} layer={e.layer}")
            if e.shapes:
                print(f"  shapes: {e.shapes}")
            if e.values:
                print(f"  values: {e.values}")


# 全局单例
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Optional[Tracer]:
    """获取全局追踪器"""
    return _global_tracer


def set_tracer(tracer: Optional[Tracer]):
    """设置全局追踪器"""
    global _global_tracer
    _global_tracer = tracer


def trace(event: str, **kwargs):
    """快捷函数：记录到全局追踪器"""
    if _global_tracer is not None:
        _global_tracer.log(event, **kwargs)


def trace_context(step: int = None, layer: int = None, block_idx: int = None):
    """快捷函数：设置全局上下文"""
    if _global_tracer is not None:
        _global_tracer.set_context(step=step, layer=layer, block_idx=block_idx)
