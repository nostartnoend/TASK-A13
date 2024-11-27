import spacy
import networkx as nx
import plotly.graph_objects as go



# 加载 spaCy 的英文模型
nlp = spacy.load('en_core_web_sm')

def read_txt_file(file_path):
    """读取 TXT 文件并返回文档列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def extract_relationships(documents):
    relationships = []
    for doc in documents:
        spacy_doc = nlp(doc)
        for sent in spacy_doc.sents:
            for token in sent:
                # 查找动词与其主语和宾语之间的关系
                if token.dep_ in ('ROOT', 'VERB'):
                    subject = [w for w in token.subtree if w.dep_ == 'nsubj']
                    object_ = [w for w in token.subtree if w.dep_ in ('dobj', 'pobj')]
                    for subj in subject:
                        for obj in object_:
                            relationships.append((subj.text, token.lemma_, obj.text))
    return relationships

# 从文本文件读取内容
file_path = r'C:\Users\37450\PycharmProjects\pythonProject9\output2\entities.txt'  # 替换为你的文本文件路径
documents = read_txt_file(file_path)

# 抽取关系
relationships = extract_relationships(documents)

print("\n抽取的关系：")
for rel in relationships:
    print(f'实体1: {rel[0]}, 关系: {rel[1]}, 实体2: {rel[2]}')


# 创建一个空图
G = nx.Graph()

# 创建边
for rel in relationships:
    subj, rel_type, obj = rel
    G.add_edge(subj, obj, relation=rel_type)

# 定义节点坐标
pos = nx.spring_layout(G, k=1.0)
x = [pos[n][0] for n in G.nodes()]
y = [pos[n][1] for n in G.nodes()]

# 提取边的坐标和标签
edge_x = []
edge_y = []
edge_text = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]  # 起点坐标
    x1, y1 = pos[edge[1]]  # 终点坐标
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)  # None 作为空隔
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    edge_text.append(f"{edge[0]} {edge[2]['relation']} {edge[1]}")  # 添加边的关系文本

# 创建节点和边的图形对象
node_trace = go.Scatter(
    x=x, y=y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="bottom center",
    marker=dict(color='lightblue', size=20, line=dict(width=2))
)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='text',
    text=edge_text,
    mode='lines'
)

# 创建图形布局
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='知识图谱',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

# 显示图形
fig.show()