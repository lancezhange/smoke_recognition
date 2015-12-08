# 统计 python 代码行数
 find . -name "*.py" | xargs grep -v "^$" | wc -l
