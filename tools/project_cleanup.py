# -*- coding: utf-8 -*-
import os
import shutil
import fnmatch
from pathlib import Path

class AegisJanitor:
    """通用项目净化专家 (V2.1 - Fix Import)"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        
        # 1. 绝对保护名单 (严禁触碰)
        self.immune_folders = ['src', 'services', 'configs', 'knowledge', 'models', 'utils', 'tools', '.git']
        self.immune_files = ['4.py', '.gitignore', 'pytest.ini']
        
        # 2. 垃圾模式名单
        self.garbage_patterns = [
            '*.bak', 
            '__pycache__', 
            '.pytest_cache', 
            '*.pyc', 
            '.ipynb_checkpoints',
            'desktop.ini',
            'Thumbs.db'
        ]

    def _is_immune(self, path: Path):
        """判定路径是否在保护名单内"""
        if path.suffix.lower() == '.md':
            return True
        if path.name in self.immune_files:
            return True
        if path.name.startswith('run_') and path.suffix == '.py':
            return True
        return False

    def cleanup(self):
        print(f"🛡️ [AEGIS JANITOR] 开始深度净化: {self.project_root}")
        count = 0

        # --- 步骤 A: 全量清理系统垃圾与备份 ---
        for root, dirs, files in os.walk(self.project_root):
            # 处理目录
            for d in list(dirs):
                if any(fnmatch.fnmatch(d, p) for p in self.garbage_patterns):
                    target = Path(root) / d
                    try:
                        shutil.rmtree(target)
                        count += 1
                        dirs.remove(d) 
                    except: pass

            # 处理文件
            for f in files:
                if any(fnmatch.fnmatch(f, p) for p in self.garbage_patterns):
                    target = Path(root) / f
                    if not self._is_immune(target):
                        try:
                            target.unlink()
                            count += 1
                        except: pass

        # --- 步骤 B: 净化 AAA 隔离区 ---
        aaa_dir = self.project_root / "AAA"
        if aaa_dir.exists():
            for item in aaa_dir.iterdir():
                if item.is_file() and not self._is_immune(item):
                    item.unlink()
                    print(f"  [AAA-Purge] 已移除探针: {item.name}")
                    count += 1
                elif item.is_dir() and item.name in ["imgs", "logs"]:
                    for sub_item in item.iterdir():
                        try:
                            if sub_item.is_file(): sub_item.unlink()
                            elif sub_item.is_dir(): shutil.rmtree(sub_item)
                        except: pass
                    print(f"  [AAA-Reset] 已重置归档区: {item.name}/")

        print(f"\n✨ [Janitor] 任务完成。处理了 {count} 个对象。项目已达到“无菌”状态。")

if __name__ == "__main__":
    janitor = AegisJanitor()
    janitor.cleanup()
