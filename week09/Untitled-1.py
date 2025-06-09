# %%
import time 
from functools import reduce
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import corpus2csc, corpus2dense, Sparse2Corpus

import pyLDAvis
import pyLDAvis.gensim_models

# %%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. 把你的字型檔路徑放到變數裡
font_path = "./font/TaipeiSansTCBeta-Regular.ttf"

# 2. 將字型加到 Matplotlib 的字型清單裡
fm.fontManager.addfont(font_path)

# 3. 取出這個字型的「內部名稱」（FontProperties.get_name() 會拿到 ttf 裡面定義的字型名稱）
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()   # 例如會得到 "Taipei Sans TC Beta" 或類似的字型名稱

# 4. 把全域 rcParams 改掉，讓 sans-serif 第一順位就是這個名稱
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name]  # 以剛剛抓到的名稱為準

# 5. 讓負號正常顯示
plt.rcParams['axes.unicode_minus'] = False

# 測試一下
x = [1, 2, 3]
y = [2, 4, 3]
plt.plot(x, y)
plt.title("全域用 TaipeiSansTCBeta 字型顯示中文")
plt.xlabel("橫軸(時間)")
plt.ylabel("縱軸(數值)")
plt.show()


# %% [markdown]
# ## 2. 資料前處理

# %%
udn = pd.read_csv("raw_data/student_2_5_8.csv")  # 匯資料
udn.head(3)

# %%
# 移除網址格式
# 只留下中文字
udn["artContent"] = udn["artContent"].str.replace("(http|https)://.*", "", regex=True)
udn["artTitle"] = udn["artTitle"].str.replace("(http|https)://.*", "", regex=True)
udn["artContent"] = udn["artContent"].str.replace("[^\u4e00-\u9fa5]+", "", regex=True)
udn["artTitle"] = udn["artTitle"].str.replace("[^\u4e00-\u9fa5]+", "", regex=True)
udn['content'] = udn['artContent']


udn = udn.loc[:,["artTitle", "content"]]  # 文章內容 文章連結
udn.head()

# %%
# invalid_idx = udn[udn["artDate_parsed"].isna()].index
# print("以下列的 artDate 解析失敗：")
# print(invalid_idx)
# print(udn.loc[invalid_idx, "artDate"].unique())

# %%
# 設定繁體中文詞庫
jieba.set_dictionary("./dict/dict.txt.big")

# 新增stopwords
# jieba.analyse.set_stop_words('./dict/stop_words.txt') #jieba.analyse.extract_tags才會作用
with open("./dict/stop_words.txt", encoding="utf-8") as f:
    stopWords = [line.strip() for line in f.readlines()]

# 設定斷詞 function
def getToken(row):
    seg_list = jieba.cut(row, cut_all=False)
    seg_list = [
        w for w in seg_list if w not in stopWords and len(w) > 1
    ]  # 篩選掉停用字與字元數大於1的詞彙
    return seg_list

udn["words"] = udn["content"].apply(getToken)
udn.head()

# %% [markdown]
# ## LDA 主題模型

# %% [markdown]
# 將斷詞後的`doc['words']`轉換成list

# %%
docs = udn['words'].to_list()
docs[0]

# %% [markdown]
# 建立並過濾詞彙表（dictionary），只保留特定條件的詞彙

# %%
dictionary = Dictionary(docs)

dictionary.filter_extremes(no_below=5, no_above=0.99)
print(dictionary)

# %% [markdown]
# 參數說明：
# - no_below=5	出現在少於 5 篇文章中的詞會被移除
# - no_above=0.99	出現在超過 99% 文件中的詞會被移除

# %%
for idx, (k, v) in enumerate(dictionary.token2id.items()):
    print(f"{k}: {v}")
    if idx > 10:
        break

# %% [markdown]
# 將斷詞結果建構語料庫(corpus)之後，利用語料庫把每篇文章數字化。<br>
# 每個詞彙都被賦予一個 ID 及頻率(word_id，word_frequency)。<br>
# 
# 舉例來說：<br>
# 第一篇文章數字化結果為：corpus[600]:[(2, 2), (6, 1), (20, 2), .... ]，element 為文章中每個詞彙的 id 和頻率。<br>
# 代表：'世界'出現2次、'之戰'出現一次...以此類推

# %%
pprint(" ".join(udn['words'].iloc[600]))

# %% [markdown]
# 第600篇文章的前十個詞彙的語料庫ID和頻率

# %%
dictionary.doc2bow(udn['words'].iloc[600])[:10]

# %% [markdown]
# #### 將docs轉換成BOW形式
# - 把每篇文件的 token list 轉換成一組 (token_id, count) 的 list

# %%
# 建立 Bag-of-words 作為文章的特徵表示
# 用 gensim ldamodel input 需要將文章轉換成 bag of words 
corpus = [dictionary.doc2bow(doc) for doc in docs]

# %% [markdown]
# **4.2 開始訓練 LDA topic model**
# 
# + 參數說明：
#     + corpus = 文檔語料庫
#     + id2word = 詞彙字典
#     + num_topics = 生成幾個主題數
#     + random_state = 固定亂數值，每次的模型結果會一樣
#     + iteration = 每個文章訓練的次數，可以設定高一點讓模型收斂
#     + passes(epoch) = 整個 corpus 訓練模型的次數
#     + alpha = 文章主題分佈
#     + eta = 主題字分佈
# 
# 模型參數沒有一個絕對的答案，同學們應該**使用相同的資料**，嘗試做參數上的調整，進而比較出較佳的模型結果。

# %%
ldamodel = LdaModel(
    corpus=corpus, 
    id2word=dictionary, # 字典
    num_topics=4, # 生成幾個主題數
    random_state=2025, # 亂數
)

# %% [markdown]
# **4.3 查看 LDA 的4個主題，每個主題中最重要的10個詞彙** <br>

# %%
ldamodel.print_topics()

# %% [markdown]
# ## 視覺化呈現
# 
# LDAvis 是我們經常會使用的視覺化工具，目的為幫助我們解釋主題模型中，在我們建構好主題模型得到 θ(文件的主題分佈) 跟 φ(主題的字分佈)，透過 pyLDAvis 將主題降維成二維，以網頁的形式供我們查看。
# 
# + 圓圈數量代表主題數量，有幾個主題就會有幾個圓圈
# + 圓越大代表 document 越大
# + 右邊可以看到主題的字分佈
# + 右上幫有一個 bar 調整 lambda：當 lambda=1 也就是代表本來的字分佈 φ，將 lambda 縮越小可以看到越唯一的字，好的分佈是 φ 高且唯一，因此我們要在這兩者間取平衡
# + 圓心越相近，代表主題會越相似；反之，圓心分越開代表主題有唯一性<br>
#   --> 假設詞彙本來有 100 字，維度應該是 100，假如本來維度接近(相近)的話，降維後也會接近(相近)
# 
# 以下用主題數 8 來做 LDAvis 的結果範例

# %%
best_model = LdaModel(
    corpus = corpus,
    num_topics = 4,
    id2word=dictionary,
    random_state = 2025,
    passes = 5 # 訓練次數
    )

# %%
pyLDAvis.enable_notebook()
p = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary)
p

# %% [markdown]
# 可以看到(7,8)、(3,6)和(2,5)很相近，試試看跑5個主題

# %%
pyLDAvis.save_html(p, "lda_zh.html")


