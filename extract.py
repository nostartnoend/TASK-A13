import os
import string
import nltk
import gensim
import gensim.corpora as corpora
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# 配置 NLTK
nltk.download('punkt')
nltk.download('stopwords')


# 步骤 1: 读取所有文本文件
def read_all_files(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts[filename] = file.read()
    return texts


# 步骤 2: 文本预处理函数
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


# 步骤 3: 提取主题
def extract_topics(paper_texts, num_topics=5):
    preprocessed_papers = {file_name: preprocess_text(text) for file_name, text in paper_texts.items()}

    # 创建字典和语料库
    dictionary = corpora.Dictionary(preprocessed_papers.values())
    corpus = [dictionary.doc2bow(preprocessed_papers[file_name]) for file_name in preprocessed_papers]

    # 训练LDA模型
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topics = {}
    for file_name in preprocessed_papers:
        bow = dictionary.doc2bow(preprocessed_papers[file_name])
        topic_distribution = lda_model.get_document_topics(bow)
        topics[file_name] = topic_distribution
    return topics, lda_model


# 步骤 4: 提取关键术语
def extract_keywords(paper_texts):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    documents = list(paper_texts.values())
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.todense().A
    keywords = {}

    for i, doc in enumerate(documents):
        feature_index = np.argsort(dense_matrix[i])[-10:]  # 获取最高的10个关键词
        keywords[list(paper_texts.keys())[i]] = [feature_names[j] for j in feature_index]

    return keywords


# 步骤 5: 实体识别
def extract_entities(paper_texts):
    nlp = spacy.load('en_core_web_sm')
    entities_by_paper = {}

    for file_name, text in paper_texts.items():
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entities_by_paper[file_name] = entities

    return entities_by_paper

def write_results_to_file(topics, keywords, entities, output_folder):
    # 如果输出文件夹不存在，则创建它
    os.makedirs(output_folder, exist_ok=True)

    topic_file_path = os.path.join(output_folder, 'topics.txt')  # 输出主题文件
    keywords_file_path = os.path.join(output_folder, 'keywords.txt')  # 输出关键词文件
    entities_file_path = os.path.join(output_folder, 'entities.txt')  # 输出实体文件

    with open(topic_file_path, 'w', encoding='utf-8') as topic_file:
        topic_file.write("提取的主题：\n")
        for file_name, topic_distribution in topics.items():
            topic_file.write(f"\n论文: {file_name}\n")
            for topic_id, prob in topic_distribution:
                topic_file.write(f"  主题 {topic_id}: {prob:.4f}\n")

    with open(keywords_file_path, 'w', encoding='utf-8') as keywords_file:
        keywords_file.write("\n提取的关键术语：\n")
        for file_name, key_terms in keywords.items():
            keywords_file.write(f"\n论文: {file_name}\n")
            keywords_file.write("  关键术语: " + ", ".join(key_terms) + "\n")

    with open(entities_file_path, 'w', encoding='utf-8') as entities_file:
        entities_file.write("\n提取的实体：\n")
        for file_name, entity_list in entities.items():
            entities_file.write(f"\n论文: {file_name}\n")
            entities_file.write(
                "  识别到的实体: " + ", ".join([f"{text} ({label})" for text, label in entity_list]) + "\n")

        # 使用这个函数来写入结果


folder_path = 'output'  # 论文转换后的文本文件夹路径
output_folder = 'output2'  # 请替换为希望输出的文件夹名称

paper_texts = read_all_files(folder_path)

# 提取主题
topics, lda_model = extract_topics(paper_texts)

# 提取关键术语
keywords = extract_keywords(paper_texts)

# 提取实体
entities = extract_entities(paper_texts)

write_results_to_file(topics, keywords, entities, output_folder)

print(f"结果已保存到 {output_folder}")