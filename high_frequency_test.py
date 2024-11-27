import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import string

# 下载 nltk 所需的资源
nltk.download('punkt')
nltk.download('stopwords')

# 读取文本文件
input_file_path = 'processed_texts.txt'  # 替换为你的文本文件路径
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 文本预处理
# 1. 转换为小写
text = text.lower()

# 2. 移除标点符号
text = text.translate(str.maketrans('', '', string.punctuation))

# 3. 分词
words = nltk.word_tokenize(text)

# 4. 移除停用词
stop_words = set(stopwords.words('english'))  # 选择适当语言的停用词
filtered_words = [word for word in words if word not in stop_words]

# 5. 生成 n-gram，提取双词组（bigrams）
bigrams = ngrams(filtered_words, 3)

# 6. 统计词组频率
bigram_counts = Counter(bigrams)

# 7. 提取高频词组
most_common_bigrams = bigram_counts.most_common(10)  # 获取前10个高频词组

# 输出结果到新的文本文件
output_file_path = 'high_frequency_bigrams3.txt'  # 输出文件路径
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for bigram, frequency in most_common_bigrams:
        output_file.write(f'{" ".join(bigram)}: {frequency}\n')

print(f'高频词组已保存到 {output_file_path}')