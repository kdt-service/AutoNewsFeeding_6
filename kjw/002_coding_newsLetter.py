import os
    # os는 파이썬 내장 모듈 중 하나로, 운영 체제와 관련된 작업을 수행할 수 
    # 있도록 기능을 제공. os 모듈을 사용하여 파일 시스템 경로를 조작하고, 
    # 파일 및 디렉토리 생성, 삭제, 이동 등의 작업을 수행할 수 있습니다. 
    # 또한, os 모듈은 운영 체제와 관련된 정보를 확인할 수 있는 함수들을 제공.
    # os 모듈은 파이썬 인터프리터에 내장되어 있으므로 별도의 설치가 필요 없슴.
from os.path import basename
    # os.path 모듈에서 basename 함수를 가져오는 구문. basename 함수는 ' 파일경로 '에서 
    # ' 파일이름 '만 추출하기 위해 사용. 예를 들어, /home/user/data/file.txt 에서 
    # basename 함수를 사용하면 ' file.txt '라는 파일만 추출할 수 있습니다.
    # 다음 코드는 basename 함수의 사용 예시다.
    # from os.path import basename
    # path = "/home/user/data/file.txt"
    # filename = basename(path)
    # print(filename) # "file.txt"만 출력될 것임.
    # 따라서 이 구문을 사용하면 os.path.basename 대신 basename만 사용하여 
    # 파일경로에서 파일이름만 추출할 수 있다. 
import codecs
    # 'utf-8'같은 인코딩을 사용하기 위해서 codecs를 불러온다.
    # codecs 모듈은 Python에서 다양한 인코딩을 지원하기 위해 제공되는 모듈.
    # 텍스트 파일을 읽거나 쓸 때 인코딩과 디코딩을 수행하고, 
    # 다양한 인코딩 형식을 지원. codecs 모듈에서 제공하는 
    # open() 함수는 일반 open() 함수와 유사하지만, 
    # 파일을 특정 인코딩 형식으로 읽거나 쓸 수 있다. 예를 들어, 
    # codecs.open(file_path, "r", "utf-8") 함수를 사용하면 
    # file_path 경로에 있는 파일을 UTF-8 인코딩 형식으로 열고, 
    # 해당 파일에서 읽거나 쓸 수 있다. 따라서, 위 코드에서 codecs 모듈은 
    # 여러가지 파일 형식(text/html등)을 UTF-8 인코딩으로 열어서 쓰기 위해 
    # 사용된다. 이렇게 함으로써, 인코딩 문제를 방지할 수 있다.
    # 참고로 codecs.open(file_path, "r", "utf-8")에서 매개변수 file_path는
    # ' 파일의 경로 '다. 코딩에서 os.path.join() 함수를 사용하여 
    # 현재 스크립트 파일이 위치한 디렉토리와 파일 이름을 결합하여 
    # 파일 경로를 생성합니다. 이를 사용하면 "상대경로" 대신 
    # "절대경로"를 사용하여 파일을 불러올 수 있다. "r"은 파일을 읽기 모드로
    # 열겠다는 매개변수다. "r"모드로 파일을 열면 파일 내용을 읽을 수 있다.
    # "r"자리에 "w"를 쓰면 쓰기 모드, "a"를 쓰면 추가모드로 파일을 연다.
    # "w"의 경우 파일을 열때 파일내용을 모두 지워버린다. "a"모드로 파일을
    # 열면 파일의 맨 마지막 끝에 새로운 내용을 추가하는 것이다.     
import pandas as pd
from konlpy.tag import Mecab
    # konlpy는 한국어 자연어처리를 위한 라이브러리로서 Mecab은 konlpy에서
    # 제공하는 형태소 분석기이다. 일본어와 함께 개발된 것이다. Mecab은 한국어 형태소
    # 분석에서 가장 높은 정확도를 보이며, 처리속도가 빠르다. Mecab은 각 형태소의
    # 품사 정보와 함께 형태소를 분석하여 반환하며, 이 정보를 이용하여 문장에서
    # 의미있는 단어들을 추출하는 자연어처리 작업에 이용한다.
from gensim import corpora, models
    # gensim도 자연어처리를 위한 라이브러리이다. corpus(말뭉치)를 다루는 모듈로서
    # corpora를 설치해야 하고, corpora가 말뭉치를 생성해서 말뭉치에 포함되어 있는
    # 단어의 출현빈도를 계산하고, 각 단어마다 고유한 ID를 매핑시키는 작업을 수행한다.
    # models는 gensim에서 지원하는 LDA(잠재디리클레할당), LSA(잠재의미분석)등
    # 토픽(topic) 모델링 알고리즘과 Word2Vec등 임베딩 알고리즘을 제공한다. models를
    # 이용해서 토픽분류 및 문서요약을 위한 순위매김(rank)을 할 수 있다.
from nltk.tokenize import sent_tokenize
    # sent_tokenize함수를 사용해서 문서를 문장 단위로 분리할 수 있다.
    # sent_tokenize함수는 구두점, 공백, 대문자 등을 이용하여 문장의 
    # 구분점 경계를 판단하며, 입력된 문서를 문장 단위로 쪼개면
    # 리스트 형태로 반환을 한다. sent_tokenize함수는 영어를 포함한
    # 다양한 언어를 지원할 수 있으며, 특히 영어 자연어처리에 많이 쓰인다.
    # nltk(Natural Language Tool kit)는 자연어처리를 위한 라이브러리이다.
    # 영어 text처리를 위한 도구와 데이터셋을 제공하는데,,, 분석, 토큰화
    # 태깅, 구문의 분석, 기계학습 등 기능을 제공한다. nltk를 이용해서
    # 텍스를 분석하고, 감성분석을 진행할 수 있으며, 텍스트 분류까지
    # 자연어처리 작업을 수행할 수 있다. NLTK는 자유소프트웨어로서 오픈소스
    # 라이브러리에 속한다.
from sklearn.metrics.pairwise import cosine_similarity
    # cosine_similarity는 벡터간의 유사한 정도를 계산하는 함수이다.
    # 코사인 유사도는 두 벡터간의 방향이 얼마나 유사한지? 나타내는 지표로
    # 벡터의 내적(dot product)을 이용해서 계산한다. 벡터간의 코사인
    # 유사도는 -1부터 1까지 범위이며 1에 가까울수록 유사도가 높은 것이다.
    # cosine_similarity(X, Y=None, dense_output=True) 함수는
    # X와 Y의 코사인 유사도를 계산하여 반환한다. X와 Y는 모두 2차원
    # 배열(또는 희소 행렬)이며, 각 행은 벡터를 나타낸다.
    # dense_output 매개변수는 결과를 밀집(dense)행렬로 반환할 것인지?
    # 희소(sparse)행렬로 반환할 것인지를 결정한다. 기본값은 True이다.
    # dense_out을 설정하지 않으면 기본(True)으로 입력되므로
    # 밀집행렬이 출력되는데, 이렇게 설정한 이유는 문서를 분석하는 과정에서는 
    # 매우 많은 단어가 사용되고, 이를 하나하나 다루기에는 시간과 메모리 
    # 측면에서 비효율이 발생되므로 dense_output 매개변수를 기본값 True로 
    # 설정하여 밀집행렬로 출력시켜서 메모리 사용을 높이고 따라서 속도를
    # 높여서 효율적인 처리를 유도한다. 밀집행렬은 모든 원소가 0이 아닌 
    # 값으로 채워진 행렬을 의미하며, 메모리 사용이 높아집니다. 
    # 하지만 모든 단어에 대한 벡터가 포함되므로 계산 과정에서 빠르고 
    # 효율적입니다. 따라서 ' 문서 분석에서는 밀집행렬이 유리 '하다.
    # 참고로 cosine_similarity 함수의 dense_output 매개변수는 
    # 출력형식을 설정하는 옵션이다.
    # dense_output=True : 두 개의 밀집(dense) 배열을 출력.
    # 이 경우 반환값은 2차원 배열로, 각 행이 입력 배열 중 하나와 
    # 유사도를 나타낸다.
    # dense_output=False : 두 개의 희소(sparse) 행렬을 출력.
    # 이 경우 반환값은 (n_samples_X, n_samples_Y) 모양의 
    # 희소행렬(sparse matrix)이다. 메모리를 절약할 수 있다.
    # sklearn.metrics.pairwise는 scikit-learn 라이브러리에서 
    # 제공하는 ' 거리 ',  ' 유사도 ' 등을 계산하는 모듈. 
    # 이 모듈을 사용하여 각 데이터 샘플 간의 거리, 
    # 유사도 등을 측정하고, 이를 바탕으로 클러스터링, 
    # 유사도 기반 검색 등의 작업을 수행. pairwise_distances 함수를
    # 비롯해 다양한 함수들이 제공되며, 이들은 numpy 배열 형태로 
    # 입력된 데이터에 대한 거리, 유사도 등을 계산하여 반환합니다. 
    # cosine_similarity 함수는 이 모듈에서 가장 자주 사용되는 함수 중 하나임.
    # numpy배열은 동일한 데이터 타입(숫자/문자열 등 모든 종류의 데이터)을
    # 원소로 가져야 하며, 다차원 배열을 처리할 수 있다. 
    # 리스트와 numpy 배열의 가장 큰 차이점은 내부 데이터를 
    # 저장하는 방식이다. 리스트는 연속된 메모리 공간에 저장되지 않고, 
    # 포인터로 구성된 구조체가 불연속적으로 메모리 공간에 
    # 분산되어 저장되는 반면 numpy 배열은 메모리 공간에 연속적으로 
    # 데이터를 저장하므로 더 빠르고 효율적인 연산이 가능하다.
    # 또한, numpy 배열은 원소들이 모두 같은 자료형을 갖기 때문에 연산시
    # 데이터 형 변환에 따른 오버헤드가 발생하지 않는다. 
    # 또한 리스트보다 많은 연산을 지원하며, 많은 수의 데이터를 다루기에도 
    # 메모리를 효율적으로 관리할 수 있다. 따라서, 숫자 연산을 다루는 경우라면,
    # numpy 배열을 사용하는 것이 효율적이다. 리스트는 원소가 다른 데이터 
    # 형태일 때 사용할 수 있는 것이 장점이다.    
from sklearn.feature_extraction.text import TfidfVectorizer
    # TfidfVectorizer 클래스는 ' 텍스트 데이터를 벡터화 '하는 데 사용. 
    # 이 클래스는 TF-IDF (Term Frequency-Inverse Document Frequency) 방법을
    # 사용하여 텍스트 데이터에서 단어의 중요성을 계산. 
    # 각 문서에서 단어가 얼마나 자주 나타나는지와 전체 문서에서 
    # 단어가 얼마나 일반적인지를 고려하여 가중치를 부여. 이를 통해 
    # 비슷한 문서간에 높은 유사성을 가진 단어는 높은 가중치를 부여함. 
    # 따라서 TfidfVectorizer를 사용하여 텍스트 데이터를 ' 벡터화 '하면 
    # 각 문서가 고유한 벡터 형태로 표현되어 머신러닝 알고리즘에 적용할 수 있다.

# HTML 파일에 작성할 내용을 저장할 변수를 생성합니다.
html_content = ""

# pandas 라이브러리를 이용해 CSV 파일을 불러옵니다.
file_path_1 = os.path.join(os.path.dirname(__file__), 'clean_IT_news.csv')
df = pd.read_csv(file_path_1).head(15)
# os.path.dirname(file)은 현재 스크립트 파일이 위치한 디렉토리를 반환하고,
# os.path.join() 함수를 사용하여 파일 경로를 생성합니다.
# 이를 사용하면 ' 상대경로 ' 대신 ' 절대경로 '를 사용하여 파일을 불러올 수 있습니다.

# 한국어 불용어 txt파일로 불용어를 처리
# 출처 : https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a#file-stopwords-ko-txt
file_name_2 = '230406_송세영님_stop_word.txt'
file_path_2 = os.path.join(os.path.dirname(__file__), file_name_2)
with open(file_path_2, 'r', encoding='utf-8') as f:
    korean_stop_words = set(line.strip() for line in f)


# 명사와 동사 품사만 추출한 후에는, 이 추출한 단어들 중에서 불용어(stopwords)를 제외한 단어들을 리스트에 반환해야 합니다. 이러한 작업은 preprocess() 함수에서 이루어집니다.
# 그리고 이제 remove_sources() 함수를 이용해서 불용어(txt 파일)에 없는 불필요한 문구를 제거합니다. 이제 이 함수를 통해 처리된 본문 기사의 내용을 가지고 요약문 생성을 위한 lda_summarize_v2() 함수를 호출할 수 있습니다.
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
        # 두번쨰부터 summary에 문장과 다음에 들어갈 문장이 유사도가 높으면 같은

        # 그 외의 경우 추가
        else:
            summary.append((sentence, score))

    summary_sentences = [sentence[0] for sentence in summary]
    return summary_sentences

# HTML 테이블 생성을 위한 시작 태그를 추가합니다.
html_content += "<table>"
html_content += "<tr><th>번호</th><th>기사 제목</th><th>기사 요약문</th></tr>"

#데이터프레임의 각 행마다 처리합니다.
for index, row in df.iterrows():
    content = row['content']
    summary = lda_summarize_v2(content, num_topics=3, num_words=10, num_sentences=num_sentences)
    # HTML 테이블에 출력할 내용을 추가합니다.
    html_content += f"<tr><td>{index+1}</td><td>{row['title']}</td><td>"
    for i in range(num_sentences):
    # summary 리스트에서 i번째 요약문을 html_content에 추가합니다.
        html_content += f"{summary[i]} "
    html_content += "</td></tr>"
    html_content += "<tr><td></td><td></td><td></td></tr>"

# HTML 테이블을 닫습니다.
html_content += "</table>"

# HTML 파일에 저장합니다.
file_path_3 = os.path.join(os.path.dirname(__file__), 'output_test_01.html')
with codecs.open(file_path_3, "w", "utf-8") as file:
    file.write(html_content)

    # 만약, 경로를 다른 디렉토리로 지정하려면, 예를 들어 실행되는 파이썬 파일의
    # 상위 디렉토리에 result 디렉토리를 만들고 그 안에 output_test_01.html 파일을 생성하려면 아래와 같이 코드를 수정할 수 있다.
    #file_path_3 = os.path.join(os.path.dirname(file), '..', 'result', 'output_test_01.html')
    #with codecs.open(file_path_3, "w", "utf-8") as file:
    #    file.write(html_content)


###########################################################################################################
################################# 출력된 HTML 파일을 뉴스레터로 보내는 코딩##################################
###########################################################################################################

import smtplib
    # 파이썬에서 이메일을 보내기 위해서 불러와야 한다. SMTP(Simple Mail Transfer 
    # Protocol)을 이용해서 이메일을 보내는데 사용한다.
import email
    # 파이썬에서 이메일 메시지를 다루기 위해서 불러온다. 이메일 모듈은
    # 이메일 메시지 작성, 헤더와 본문의 설정, MIME(Multipurpose Internet Mail
    # Extensions)형식 으로 변환해 준다. MIME형식으로 변환해야 전송이 가능하다.
from email.mime.multipart import MIMEMultipart
    # MIMEMultipart 클래스를 이용해서 이메일 메시지를 생성. MIMEMultipart클래스는
    # 첨부파일이 있는 이메일이나 HTMl 같이 이메일 메시지(이하 '이메일')가
    # 여러 부분(multipart)으로 구성되어 있을때 사용한다. 각 부분을 MIME타입으로
    # 지정해 놓고 각 부분의 내용을 추가하거나, 첨부파일을 추가할때 매우 용이하다.
    # 즉, MIMEMultipart클래스(설계도)로 이메일 이라는 객체를 생성하고,
    # 변수(보통 msg를 많이 사용)에 할당해 준 다음 송신자, 수진자, 제목 등의 정보를
    # 지정하는 것이다. 이메일의 ' 본문 부분은 MIMEText클래스 '를 불러와서 생성시킨다.
    # msg.attach()메서드를 사용해서 이메일 본문 즉, ' text '를 ' 이메일 객체 '에 
    # 추가하는 방식이다. 그리고, 첨부파일의 내용을 읽어들여서 ' MIMEText 객체 혹은
    # MIMEBase 객체 '로 만들고, add_header()메서드를 사용해서 첨부파일 정보를
    # 지정하여 이메일 객체에 추가해 주는 것이다.
from email.mime.text import MIMEText
    # 일반 text 데이터를 MIME형식으로 변환하는데 사용. MIMEText클래스는 이메일 본문
    # 내용을 작성시 사용. MIMEText클래스를 사용하여 MIMEText객체의 본문내용('body') 설정,
    # MIME타입('subtype') 설정, 인코딩('charset')을 설정할 수 있다.
    # MIMEText객체는 MIMEMultpart객체와 함께 사용된다.
from email.mime.base import MIMEBase
    # 이메일 메시지에 첨부파일(이미지/동영상/음성파일 등 다양한 형식)을 추가하기 위해 사용.
    # MIME(Multipurpose Internet Mail Extensions)는 인터넷에서 이메일을 보내는 데 사용되는
    # 프로토콜로서 이메일에 ' 텍스트/이미지/첨부파일 '을  처리하기 위해서 만들어졌다.
from email import encoders
    # 이진(binary) 데이터를 MIME 형식으로 인코딩하기 위한 함수를 제공. 
    # 이 모듈의 encode_base64() 함수는 이진 데이터를 ' Base64 인코딩 '하여 
    # ' 문자열 '로 반환. 이는 이진 데이터를 ' 텍스트 기반 ' 이메일 전송 방식에 
    # 적합하도록 인코딩하는 데 사용. 이 모듈을 이용하면 이진 파일(이미지/문서파일 등)을 
    # 이메일에 첨부할 때, 이진 데이터를 문자열로 인코딩하여 첨부할 수 있다. 더 자세하게
    # 설명을 하자면,,, 이진 데이터(binary data)는 컴퓨터에 이진수로 저장된 데이터를 
    # 말하며, 이를 인코딩(encoding)하여 문자열 형태로 변환할 수 있다. 
    # 이때 사용되는 대표적인 방식 중 하나가 ' Base64 인코딩 '이다. Base64는 8비트 
    # 이진 데이터를 6비트씩 묶어서 ASCII 문자로 변환하는 방식으로, 
    # 64개의 문자(알파벳 대소문자, 숫자, '+'와 '/')를 이용하여 표현하는 것이다. 
    # 이를 이용해 binary data를 문자열로 인코딩하여 첨부할 수 있다. 이때 encoders 
    # 모듈의 encode_base64() 함수를 사용한다. 이 함수는 MIMEBase 객체의 payload속성에 
    # 저장된 binary data를 Base64로 인코딩하여 ASCII 문자열로 변환한다. 
    # 변환된 문자열은 MIMEBase 객체의 'Content-Transfer-Encoding' 헤더의 값으로 
    # 설정되어 전송되는 것이다.
  
from email.mime.image import MIMEImage
    # 이메일 본문(body)에 기사요약으로 출력된 파일 'output_test_01.html'이 내용을 이미지로
    # 붙이기 위해서  email.mime.image모듈에서 MIMEImage클래스를 추가로 불러왔다.
from PIL import Image
    # Anaconda prompt 에서 가상환경(ex. study 혹은 study2 등)을 활성화
    # 시킨 후 conda install Pillow 를 실행한다. pip명령어를 써도 되는데
    # 전역이 아니라 내가 직접 만든 가상환경에서만 Pillow를 설치하기 위해서
    # conda명령어를 사용했슴. 가상환경에만 설치를 해서 만약 충돌이 일어나면,
    # 가상환경만 새로 만들어서 Anaconda3와 VSCode를 전부 재설치하는 
    # 번거로움 피하기 위해서임. Pillow(PIL)은 html을 이미지로 이메일 본문에
    # 보여주기 위해서 이미지 라이브러리를 설치한 것임.
import imghdr
    # imghdr 모듈은 이미지 파일의 형식을 확인하는 표준 라이브러리. 
    # 따라서 imghdr 모듈을 사용하면 이미지 파일이 어떤 형식인지 
    # 쉽게 확인. 이미지 파일의 형식을 확인하는 이유??? 해당 파일을 
    # 올바르게 처리하기 위함. 예를 들어, JPEG 형식의 이미지 파일에 
    # 대해 PNG 형식으로 처리하려고 시도하면 올바르게 처리되지 않을 
    # 가능성이 있다. 이런 경우 imghdr 모듈을 사용하여 파일 형식을 
    # 확인하고, 해당 파일 형식에 따라 올바르게 처리. 또한, 
    # 업로드한 파일을 처리할 때, 보안 상의 이유로 이미지 파일이 
    # 아닌 다른 파일을 업로드하는 것을 방지하기 위해서도 
    # imghdr 모듈을 사용하여 이미지 파일인지 확인. 코드사용 방법은
    # https://docs.python.org/ko/3/library/imghdr.html?highlight=import%20imghdr
    # 확인 가능하다.
            # if imghdr.what('test.jpg') == 'jpeg':
            #       print('test.jpg is a JPEG image')
            # else:
            #       print('test.jpg is not a JPEG image')
import base64
import io


SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
SMTP_USER = '?????????????????????????????????????'  # 각 개인의 이메일 아이디
SMTP_PASSWORD = '?????????????????????????????????'  # 각 개인이 설정한 SMTP 비밀번호 입력

to_users = ['kjy153@naver.com',  'kjy153@gmail.com'] # 수신인 주소 기입하는 곳
target_addr = ','.join(to_users)

subject = '제목: test news email by kimjongwoo from NAVER'
contents = '변경되지 않는 내용을 여기에 명시하면 됩니다. 변경될 내용은 기사요약입니다.'

# 이메일 송신인 / 수신인 / 제목 / 본문내용 생성
msg = MIMEMultipart('related')  # mixed가 아닌 related로 설정해도 별첨화일을 송부할 수 있다.

msg['From'] = SMTP_USER   #MIMEMultipart객체의 송신인 필드에 값을 넣어준 것임
msg['To'] = target_addr   #MIMEMultipart객체의 수신인 필드에 값을 넣어준 것임
msg['Subject'] = subject  #MIMEMultipart객체의 제목 필드에 값을 넣어준 것임

#text = MIMEText(contents)   # 충돌은 아니고, 겹치는 것을 피하기 위해 비활성화시켰슴
#msg.attach(text)            # 충돌은 아니고, 겹치는 것을 피하기 위해 비활성화시켰슴
    # 아래 2개의 코드 html = MIMEText(html, 'html') msg.attach(html)와 겹치는 것을 
    # 피하기 위해 위 2개의 코드를 비활성화시킴


# 이메일에 첨부할 html 파일 추가
file_name_4 = os.path.basename(file_path_3) 
    #'output_test_01.html'파일명과 경로를 ' file_path_3 '에 이미 할당해 놓았기 때문에 
    # os.path.basename()메서드로 연결시켜서 'output_test_01.html'파일명을 2번 써 줄 필요가 없도록 코딩(실수가능성차단)  
file_path_4 = os.path.join(os.path.dirname(__file__), file_name_4)
with open(file_path_4, 'r', encoding='utf-8') as f:
    file_data = f.read()

email_file = MIMEBase('text', 'html', charset='UTF-8')
email_file.set_payload(file_data.encode('UTF-8'))
encoders.encode_base64(email_file)
email_file.add_header('Content-Disposition', 'attachment', filename=file_name_4)  
msg.attach(email_file)

# 이메일 본문에 붙여넣을 html 코딩은 아래와 같다. 그림이 먼저 나오게 하기 위해서 아래 코드를
# 비활성시키고, 복사해서 맨 아래에 붙여 넣는다. 그러면 그림(이미지)이 먼저 나오고 그 다음에 기사요약이 나올 것이다.
# html = MIMEText(file_data, 'html')
# msg.attach(html)


# 이메일 본문에 이미지를 Inline으로 붙여넣기 
# Open the image file and decode it using PIL
import io
import imghdr
import base64

file_name_pic_1 = 'img_newsletter.jpg'  # 사진파일이름 저장
file_path_5 = os.path.join(os.path.dirname(__file__), file_name_pic_1)
with open(file_path_5, 'rb') as file:
    img_data = file.read()
    img = Image.open(io.BytesIO(img_data))  
img_type = imghdr.what(None, img_data)
# Create a MIMEImage object and set the Content-ID header
image = MIMEImage(img_data, _subtype=img_type)
image.add_header('Content-ID', '<image1>')
#msg.attach(image) # 이 코드를 활성화시키면 image화일이 첨부된다.


encoded_string = base64.b64encode(open(file_path_5, 'rb').read()).decode()

#html = '''<img alt="ImageLoadingErr" width="805" height="502" style="border:none;" src="data:image/jpeg;base64,{logo}" />'''.format(logo=encoded_string)
    # 위 코딩을 비활성화시키고 가시성을 높이기 위해 아래 코딩으로 변경하였다. 동일하다.
html = f"""\
<html>
  <body>
    <p>{contents}</p>
    <p>여기는 이미지가 붙을 자리입니다. 여기 아래에 이미지가 붙어야 합니다.</p>
    <p><img alt="ImageLoadingErr" width="805" height="502" style="border:none;" 
    src="data:image/jpeg;base64,{encoded_string}" /></p>
  </body>
</html>
"""
html = MIMEText(html, 'html')
msg.attach(html)  # 이 코드를 활성화하면 위의 msg.attach(text) 가 표시되지 않는다.
    # 왜냐하면, text를 MIMEText()함수를 사용하여 먼저 담아 놓았는데, 여기서 다시
    # MIMEText()함수를 다시 설정하여 html을 담았기 때문이다. 만약 
    # msg.attach(text)와 msg.attach(html) 모두 이메일에 표시하려면 두 가지 방법이 있다.
    # msg.attach(text)와 msg.attach(html)를 모두 활성화하고, msg.attach(html)에서 
    # 이전에 msg.attach(text)에서 추가한 텍스트 내용을 포함하여 HTML 코드를 작성한다. 
    # 예를 들어:
            # 첫번째 코딩 방법
            # html = f"""\
            # <html>
            #   <body>
            #     <p>{contents}</p>
            #     <p>여기는 이미지가 붙을 자리입니다. 여기 아래에 이미지가 붙어야 합니다.</p>
            #     <p><img alt="ImageLoadingErr" width="805" height="502" style="border:none;" 
            #     src="data:image/jpeg;base64,{encoded_string}" /></p>
            #   </body>
            # </html>
            # """
            # html = MIMEText(html, 'html')
            # msg.attach(html)
            ################################
            # 두번재 코딩 방법은
            # msg.attach(text)를 삭제하고, HTML 코드 내부에 contents 변수를 포함하여 이메일 본문을 작성하고, 
            # 그것을 MIMEText 객체로 만들어 msg.attach()에 추가한다.
            # html = f"""\
            # <html>
            #   <body>
            #     <p>{contents}</p>
            #     <p>여기는 이미지가 붙을 자리입니다. 여기 아래에 이미지가 붙어야 합니다.</p>
            #     <p><img alt="ImageLoadingErr" width="805" height="502" style="border:none;" 
            #     src="data:image/jpeg;base64,{encoded_string}" /></p>
            #   </body>
            # </html>
            # """
            # msg.attach(MIMEText(html, 'html'))

# 기사요약 으로 출력한 output_test_01.html파일의 내용을 기사본문에 붙이는 코딩이다.
html = MIMEText(file_data, 'html')
msg.attach(html)


smtp = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
smtp.login(SMTP_USER, SMTP_PASSWORD)
smtp.sendmail(SMTP_USER, to_users, msg.as_string())
smtp.close()

