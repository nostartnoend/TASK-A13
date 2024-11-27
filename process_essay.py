import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# 下载nltk的停用词
nltk.download('punkt')
nltk.download('stopwords')

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')


def pdf_to_text(pdf_file_path):
    """将PDF文件转换为纯文本"""
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text



def preprocess_text(text):
    """对文本进行预处理，去除停用词，标点和词形还原"""
    # 转为小写
    text = text.lower()

    # 标记化
    words = word_tokenize(text)

    # 去除停用词和标点符号
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # 词形还原
    lemmatized_words = [nlp(word)[0].lemma_ for word in filtered_words]

    return ' '.join(lemmatized_words)


def process_documents(input_directory, output_file,output_folder):
    """处理多个文档，转换为统一格式并预处理"""
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, output_file)  # 输出文件路径

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(input_directory):
            if filename.endswith(".pdf"):
                pdf_file_path = os.path.join(input_directory, filename)
                print(f"Processing: {pdf_file_path}")

                # 将PDF转换为文本
                text = pdf_to_text(pdf_file_path)

                # 预处理文本
                processed_text = preprocess_text(text)

                # 写入输出文件
                outfile.write(f"Document: {filename}\n")
                outfile.write(processed_text + "\n\n")

            # 使用示例


input_directory = r'C:\Users\37450\PycharmProjects\pythonProject9\input_directory'  # 替换为PDF文件存放的目录
output_file = 'processed_texts.txt'  # 输出文件名
process_documents(input_directory, output_file,'output')