import pandas as pd
import os
from konlpy.tag import Mecab
from gensim import corpora, models
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer    
import codecs

# HTML 파일에 작성할 내용을 저장할 변수를 생성합니다.
html_content = ""

# pandas 라이브러리를 이용해 CSV 파일을 불러옵니다.
file_path_1 = os.path.join(os.path.dirname(__file__), 'clean_IT_news.csv')
df = pd.read_csv(file_path_1).head(15)
    # os.path.dirname(__file__)은 현재 스크립트 파일이 위치한 디렉토리를 반환하고, 
    # os.path.join() 함수를 사용하여 파일 경로를 생성합니다. 
    # 이를 사용하면 ' 상대경로 ' 대신 ' 절대경로 '를 사용하여 파일을 불러올 수 있습니다.



# 한국어 불용어 txt파일로 불용어를 처리
# 출처 : https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a#file-stopwords-ko-txt

filename = '230406_송세영님_stop_word.txt'
file_path_2 = os.path.join(os.path.dirname(__file__), filename)
with open(file_path_2, 'r', encoding='utf-8') as f:
    korean_stop_words = set(line.strip() for line in f)

# mecab 형태소 분석기 사용
def preprocess(text):
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    # Mecab을 이용해서 문장에서 명사와 동사 품사만 추출한다.
    tokens = [word for word in mecab.pos(text) if word[1][0] in ['N', 'V']]
    # 추출한 단어중 불용어를 제외한 단어를 리스트에 반환한다.
    tokens = [word for word, pos in tokens if word not in korean_stop_words]
    return tokens

def remove_sources(text, sources=['출처=IT동아', '사진=트위터 @DylanXitton']):
    # 불용어 txt파일에 없는 불필요한 문구를 제거
    for source in sources:
        text = text.replace(source, '')
    return text

# TF-IDF를 이용하여 문장 유사도계산
def sentence_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    # 두 문장을 입력받아 각각 TF-IDF 벡터 생성하고 행렬형태로 저장
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    # 코사인유사도 계산 후 리턴
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# 본문기사 요약
num_sentences = 3

def lda_summarize_v2(text, num_topics=3, num_words=10, num_sentences=num_sentences, similarity_threshold=0.6):
    # 요약문을 생성하는 함수 정의, text는 요약 대상 텍스트, num_topics는 토픽 수, 
    # num_words는 토픽당 키워드 수, num_sentences는 요약문에 포함될 문장 수입니다.
    # similarity_threshold는 문장 유사도
    text = remove_sources(text)
    # 불필요 문구제거
    sentences = sent_tokenize(text)
    # 문장단위로 나누기
    tokenized_sentences = [preprocess(sentence) for sentence in sentences]
    # 문장 전처리 후 토큰화
    dictionary = corpora.Dictionary(tokenized_sentences)
    # 단어를 고유ID에 매핑
    corpus = [dictionary.doc2bow(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
    # 각 단어 출연빈도를 단어ID, 단어 빈도 쌍으로 리스트 변환
    lda_model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    # Gensim LDA 모델을 생성하고 학습
    # passes는 반복수
    topic_keywords = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    # 각 토픽의 단어 분포와 해당 단어들의 점수 반환
    topic_sentences = []

    for topic, keywords in topic_keywords:
        topic_words = [word[0] for word in keywords]
        # 토픽의 키워드를 리스트로 생성
        for sentence in sentences:
            tokenized_sentence = preprocess(sentence)
            # 돌면서 문장을 전처리 + 토큰화
            if any(word in tokenized_sentence for word in topic_words):
                topic_sentences.append(sentence)
            # 문장 중 토픽키워드가 있으면 topic_sentences에 추가

    ranked_sentences = sorted([(sentence, lda_model[corpus[sentences.index(sentence)]][0][1]) for sentence in topic_sentences], key=lambda x: x[1], reverse=True)
    # 모든 topic_sentences에 대해 해당 문장이 속한 토픽의 점수를 계산하고, 점수가 높은 순서로 정렬합니다.

    summary = []
    for sentence, score in ranked_sentences:
        # ranked_sentence에서 문장과 점수를 꺼냄
        if len(summary) >= num_sentences:
            break
        # 문장길이가 예정된 요약문 길이를 넘으면 반복문 종료
        
        for chosen_sentence, i in summary:
            if sentence_similarity(sentence, chosen_sentence) > similarity_threshold:
                break
        # 처음에는 summary가 빈 리스트이기에 else문이 바로 실행
        # 두번쨰부터 summary에 문장과 다음에 들어갈 문장이 유사도가 높으면 같은 의미를 가진 내용이라 생각하고 제외

        else:
            summary.append((sentence, score))
        # 그 외의 경우 추가

    summary_sentences = [sentence[0] for sentence in summary]
    # 요약문 리스트에서 점수를 제외하고 문장만 추출
    return summary_sentences

# HTML 테이블 생성을 위한 시작 태그를 추가합니다.
html_content += "<table>"
html_content += "<tr><th>번호</th><th>기사 제목</th><th>기사 요약문</th></tr>"
for index, row in df.iterrows():
    content = row['content']
    summary = lda_summarize_v2(content, num_topics=3, num_words=10, num_sentences=num_sentences)
    # HTML 테이블에 출력할 내용을 추가합니다.
    html_content += f"<tr><td>{index+1}</td><td>{row['title']}</td><td>"
    for i in range(num_sentences):
        html_content += f"{summary[i]} "
    html_content += "</td></tr>"
    html_content += "<tr><td></td><td></td><td></td></tr>"
# HTML 테이블을 닫습니다.
html_content += "</table>"

# HTML 파일에 저장합니다.
file_path_3 = os.path.join(os.path.dirname(__file__), 'output_2.html')
with codecs.open(file_path_3, "w", "utf-8") as file:
    file.write(html_content)
    
# 만약, 경로를 다른 디렉토리로 지정하려면, 예를 들어 실행되는 파이썬 파일의 
# 상위 디렉토리에 result 디렉토리를 만들고 그 안에 output_2.html 파일을 생성하려면 아래와 같이 코드를 수정할 수 있다.
# file_path = os.path.join(os.path.dirname(__file__), '..', 'result', 'Figma_output_2.html')
# with codecs.open(file_path, "w", "utf-8") as file:
#     file.write(html_content)    
