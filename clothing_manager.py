import os
import random
import re
from typing import Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as _logger
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)

class ClothingManager:
    """衣服标签管理器"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.clothing_data: Dict[str, str] = {}
        self.categories: Dict[str, List[str]] = {}
        self.category_names: Dict[str, str] = {
            '职业装': 'professional',
            '女仆装': 'maid', 
            '学生装': 'student',
            '战斗装': 'combat',
            '礼服装': 'formal',
            '休闲装': 'casual',
            '运动装': 'sports',
            '传统装': 'traditional',
            '角色扮演': 'cosplay',
            '内衣装': 'lingerie',
            '其他装': 'other'
        }
        self.load_database()
    
    def load_database(self):
        """加载衣服数据库"""
        if not os.path.exists(self.data_file):
            _logger.warning(f"衣服数据库文件不存在: {self.data_file}")
            return
        
        try:
            self.clothing_data.clear()
            self.categories.clear()
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('-', 3)
                    if len(parts) == 4:
                        category, name, number, tags = parts

                        # 验证编号格式（应该是3位数字）
                        if number.isdigit() and len(number) == 3:
                            self.clothing_data[number] = tags

                            # 构建分类索引
                            if category not in self.categories:
                                self.categories[category] = []
                            self.categories[category].append(number)
                        else:
                            _logger.warning(f"跳过格式错误的行: {line[:100]}...")
            
            _logger.info(f"✅ 成功加载 {len(self.clothing_data)} 套衣服数据")
            _logger.info(f"📊 分类统计: {', '.join([f'{k}({len(v)})' for k, v in self.categories.items()])}")
            
        except Exception as e:
            _logger.error(f"加载衣服数据库失败: {e}")
    
    def reload_database(self):
        """重新加载数据库"""
        _logger.info("重新加载衣服数据库...")
        self.load_database()
    
    def get_clothing_by_number(self, number: str) -> Optional[str]:
        """根据编号获取衣服标签"""
        # 支持1位、2位、3位数字
        padded_number = number.zfill(3)
        return self.clothing_data.get(padded_number)
    
    def get_random_clothing(self, category: str = None) -> Optional[str]:
        """获取随机衣服"""
        if category:
            # 支持中文分类名和英文别名
            if category in self.categories:
                numbers = self.categories[category]
            else:
                # 尝试通过英文别名查找
                category_cn = None
                for cn, en in self.category_names.items():
                    if en == category.lower():
                        category_cn = cn
                        break
                
                if category_cn and category_cn in self.categories:
                    numbers = self.categories[category_cn]
                else:
                    _logger.warning(f"未找到分类: {category}")
                    return None
        else:
            numbers = list(self.clothing_data.keys())
        
        if numbers:
            number = random.choice(numbers)
            return self.clothing_data[number]
        return None
    
    def process_prompt(self, prompt: str) -> str:
        """处理提示词中的衣服标签"""
        original_prompt = prompt
        
        # 处理随机衣服
        if "随机衣服" in prompt:
            # 检查是否指定分类 - 支持多种格式
            category_patterns = [
                r'随机衣服\s+(\S+)',  # 随机衣服 职业装
                r'随机(\S+)衣服',     # 随机职业装衣服  
            ]
            
            category = None
            for pattern in category_patterns:
                match = re.search(pattern, prompt)
                if match:
                    category = match.group(1)
                    break
            
            if category:
                clothing_tags = self.get_random_clothing(category)
                if clothing_tags:
                    # 替换所有匹配的模式
                    for pattern in category_patterns:
                        prompt = re.sub(pattern, clothing_tags, prompt)
                    prompt = prompt.replace('随机衣服', clothing_tags)
                else:
                    _logger.warning(f"未找到分类 {category} 的衣服，使用完全随机")
                    clothing_tags = self.get_random_clothing()
                    prompt = prompt.replace('随机衣服', clothing_tags or '')
            else:
                clothing_tags = self.get_random_clothing()
                prompt = prompt.replace('随机衣服', clothing_tags or '')
            
            if clothing_tags:
                _logger.debug(f"随机选择衣服: {clothing_tags[:50]}...")
        
        # 处理衣服编号 - 只匹配 "衣服131" 这样的格式，避免误触
        clothing_pattern = r'衣服(0\d{2}|[1-4]\d{2}|430)\b'  # 匹配 "衣服001" 到 "衣服430"

        # 只处理第一个匹配的衣服编号
        match = re.search(clothing_pattern, prompt)
        if match:
            number = match.group(1)
            clothing_tags = self.get_clothing_by_number(number)
            if clothing_tags:
                # 替换整个 "衣服131" 为衣服标签
                prompt = prompt.replace(f"衣服{number}", clothing_tags, 1)
                _logger.debug(f"使用衣服编号 {number}: {clothing_tags[:50]}...")
        
        return prompt
    
    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return list(self.categories.keys())
    
    def get_category_range(self, category: str) -> str:
        """获取分类的编号范围"""
        if category in self.categories:
            try:
                numbers = [int(n) for n in self.categories[category]]
                return f"{min(numbers):03d}-{max(numbers):03d}"
            except ValueError as e:
                _logger.warning(f"解析编号时出错: {e}")
                _logger.debug(f"分类 {category} 的编号: {self.categories[category]}")
                return "解析错误"
        return ""
    
    def get_category_count(self, category: str) -> int:
        """获取分类的数量"""
        return len(self.categories.get(category, []))
    
    def search_clothing(self, keyword: str) -> List[Tuple[str, str, str, str]]:
        """搜索包含关键词的衣服"""
        results = []
        keyword_lower = keyword.lower()
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('-', 3)
                    if len(parts) == 4:
                        category, name, number, tags = parts
                        if keyword_lower in tags.lower() or keyword_lower in name.lower():
                            results.append((number, category, name, tags))
        except Exception as e:
            _logger.error(f"搜索失败: {e}")
        
        return results

    def get_clothing_detail(self, number: str) -> Optional[Tuple[str, str, str]]:
        """根据编号获取衣服详情"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split('-', 3)
                    if len(parts) == 4:
                        category, name, line_number, tags = parts
                        if line_number == number:
                            return (category, name, tags)
        except Exception as e:
            _logger.error(f"获取衣服详情失败: {e}")

        return None
    
    def get_statistics(self) -> str:
        """获取数据库统计信息"""
        total = len(self.clothing_data)
        stats = []
        stats.append(f"📊 衣服数据库统计")
        stats.append(f"总数量: {total} 套")
        stats.append(f"分类数: {len(self.categories)} 个")
        stats.append("")
        stats.append("分类详情:")
        
        for category, numbers in self.categories.items():
            count = len(numbers)
            range_info = self.get_category_range(category)
            stats.append(f"  {category}: {count}套 ({range_info})")
        
        return "\n".join(stats)
    
    def get_clothing_info(self, number: str) -> Optional[Tuple[str, str, str]]:
        """获取指定编号的衣服详细信息"""
        padded_number = number.zfill(3)
        
        if padded_number not in self.clothing_data:
            return None
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('-', 3)
                    if len(parts) == 4:
                        category, name, num, tags = parts
                        if num == padded_number:
                            return (category, name, tags)
        except Exception as e:
            _logger.error(f"获取衣服信息失败: {e}")
        
        return None
    
    def format_clothing_list(self, category: str, limit: int = 10) -> str:
        """格式化显示分类中的衣服列表"""
        if category not in self.categories:
            return f"未找到分类: {category}"
        
        numbers = self.categories[category][:limit]
        results = []
        results.append(f"🎀 {category} ({len(self.categories[category])}套)")
        results.append("")
        
        for number in numbers:
            info = self.get_clothing_info(number)
            if info:
                cat, name, tags = info
                # 截断过长的标签
                short_tags = tags[:60] + "..." if len(tags) > 60 else tags
                results.append(f"{number}: {name}")
                results.append(f"     {short_tags}")
                results.append("")
        
        if len(self.categories[category]) > limit:
            results.append(f"... 还有 {len(self.categories[category]) - limit} 套")
        
        return "\n".join(results)
