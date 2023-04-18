import os
    # 파일 시스템 경로를 조작, 디렉토리 생성/삭제/이동 작업 수행 
from os.path import basename
    # ' 파일경로 '에서 ' 파일이름 '만 추출.
import codecs
    # 'utf-8'같은 인코딩을 사용하기 위한 모듈     
import pandas as pd
from konlpy.tag import Mecab
    # konlpy는 한국어 자연어처리를 위한 라이브러리로서 Mecab은 konlpy에서 제공하는 형태소 분석기임.
from gensim import corpora, models
    # corpora: 단어의 출현 빈도를 계산 및 각 단어별 고유 ID를 매핑.
    # models:  LDA(잠재디리클레할당), LSA(잠재의미분석)등 토픽(topic) 및 Word2Vec등 임베딩 알고리즘(토픽분류/문서요약/순위매김)을 제공.
from nltk.tokenize import sent_tokenize
    # 문서를 문장 단위로 분리(by 구두점, 공백, 대문자 등을 이용)
from sklearn.metrics.pairwise import cosine_similarity
    # cosine_similarity는 벡터간의 유사도를 계산.(범위는 -1부터 1까지 범위이며 1에 가까울수록 유사도가 높은 것) 
from sklearn.feature_extraction.text import TfidfVectorizer
    # ' 텍스트 데이터를 벡터화 ', TF-IDF (Term Frequency-Inverse Document Frequency) 방법을 사용.

    
# HTML 파일에 작성할 내용을 저장할 변수를 생성합니다.
html_content = ""
    # 빈 문자열(empty string)로 초기화(initialize). 변수 html_content에 ""(빈 문자열)을 할당.
    
# pandas 라이브러리를 이용해 CSV 파일을 불러옵니다.
file_path_1 = os.path.join(os.path.dirname(__file__), 'clean_IT_news.csv')
df = pd.read_csv(file_path_1).head(15)

# 한국어 불용어 txt파일로 불용어를 처리
file_name_2 = '230406_송세영님_stop_word.txt'
file_path_2 = os.path.join(os.path.dirname(__file__), file_name_2)
with open(file_path_2, 'r', encoding='utf-8') as f:
    korean_stop_words = set(line.strip() for line in f)

# mecab 형태소 분석기 사용
def preprocess(text):
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    # Mecab을 이용해서 문장에서 명사와 동사 품사만 추출.
    tokens = [word for word in mecab.pos(text) if word[1][0] in ['N', 'V']]
    # 추출한 단어중 불용어를 제외한 단어를 리스트에 반환.
    tokens = [word for word, pos in tokens if word not in korean_stop_words]
    return tokens    
def remove_sources(text, sources=['출처=IT동아', '사진=트위터 @DylanXitton']):
    # 불용어 txt파일에 없는 불필요한 문구를 제거하기 위해 sources변수에 제거대상 리스트를 할당
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
# 요약문 생성 함수 정의
def lda_summarize_v2(text,                          # text는 요약 대상 텍스트 
                     num_topics=3,                  # num_topics는 토픽 수
                     num_words=10,                  # num_words는 토픽당 키워드 수
                     num_sentences=num_sentences,   # num_sentences는 요약문에 포함될 문장의 수
                     similarity_threshold=0.6):     # 유사도 설정 하이퍼파라미터(0~1사이 설정, 보통 0.6으로 설정함)
    
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
    # LDA모델 생성/학습(num_topics: 추출토픽개수, id2word:인덱스와 단어매핑, passes: 학습횟수)
    
    topic_keywords = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    # 토픽 해당 키워드 추출(num_topics: 추출토픽개수, num_words:키워드개수, formatted: False이면 리스트로 출력)
    
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

    ranked_sentences = sorted(
        [(sentence, lda_model[corpus[sentences.index(sentence)]][0][1]) for sentence in 
         topic_sentences], 
        key=lambda x: x[1], 
        reverse=True)
    # 모든 topic_sentences에 대해 해당 문장이 속한 토픽의 점수를 계산하고, 점수가 높은 순서로 정렬.

    summary = []    # 빈 리스트(empty list)를 초기화해서 summary 변수에 할당.
    
    # 아래 코딩은 기사요약을 다른 문장으로 구성하기 위해서 송세영팀장님이 추가한 코딩입니다.                
    for sentence, score in ranked_sentences:
        # ranked_sentence에서 문장과 점수를 꺼냄
    
        if len(summary) >= num_sentences:
            break
        # 문장길이가 예정된 요약문 길이를 넘으면 반복문 종료시킴(위쪽 코딩을 찾아보면 3을 num_sentences에 할당해 놓았슴)
        
        for chosen_sentence, i in summary:
            if sentence_similarity(sentence, chosen_sentence) > similarity_threshold:
                break
        # summary 초기화로 바로 아래 else문이 실행된 후, 두번째 문장부터 다음 문장과 유사도가 높으면 '중복문장'이라고 판단하여 break 실행
        # 즉, 중복문을 제거하고, 다른 문장으로 기사요약을 구성하기 위해 정세영팀장님이 추가한 코딩이다. 
        else:
            summary.append((sentence, score))

    summary_sentences = [sentence[0] for sentence in summary]
    return summary_sentences

# HTML 테이블 생성 위한 시작 태그.
html_content += "<table>"
    # html_content를 <table> 태그에 추가한다.
    
html_content += "<tr><th>번호</th><th>기사 제목</th><th>기사 요약문</th><th>URL</th></tr>"
    # 위 코드는 HTML 테이블의 헤더(row)를 구성하는 코드. <tr>은 HTML에서 행(row)을 구성하는 태그. <th>는 HTML에서 header cell을 구성.

for index, row in df.iterrows(): # 데이터프레임의 row(행)을 반복하면서 값을 row변수에 인덱스는 index변수에 저장
    content = row['content']     # row(행)을 반복하면서 각각의 row(행)에 해당하는 각각의 content컬럼(기사내용)이 변수 content 에 할당됨  
    summary = lda_summarize_v2(content, num_topics=3, num_words=10, num_sentences=num_sentences)
                                 # lda_summarize_v2함수로 기사요약. 조건인수는 num_topics(생성할토픽수), num_words(기사요약내단어수), num_sentences(기사요약시 문장의 수) 를 적용.
    url = row['url'] # url 추출. 데이터프레임의 열(컬럼)에서 url컬럼 값을 가져와서 변수 url에 할당한다. 
  
    # HTML 테이블에 출력할 내용을 추가합니다.
    html_content += f"<tr><td>{index+1}</td><td>{row['title']}</td><td>"  # '행구분 태그'의 <tr>로 시작하고 행 구분 끝</tr>은 아래 for문을 돌린후 붙여주도록 코딩 
    for i in range(num_sentences):
        html_content += f"{summary[i]} "  # num_sentences를 3으로 지정해 놓았기 때문에 기사를 순위에 따라서 3개까지 테이블(html_content)에 넣어 주는 코딩이다. 
    html_content += f"</td><td>{url}</td></tr>" # 시작이 '셀(테이블데이터) 구분 태그'의 끝이다. 즉 기사요약 문장 3개를 넣어주고 url 컬럼까지 포함시킨후 행 구분 끝</tr>을 붙여줌                   # 그리고 이어서 셀구분 태크가 열리면서 을 기사요약셀 다음셀에 넣어주고 셀구분 태크 끝이 오고 바로 행 구분 태그의 끝이 와서 행이 끝나는 것이다.
    html_content += "<tr><td></td><td></td><td></td><td></td></tr>" # 테이블 가독성을 높이기 위해서 빈 행을 추가했슴.         

# HTML 테이블을 닫습니다.
html_content += "</table>"

# HTML 파일에 저장합니다.
file_path_3 = os.path.join(os.path.dirname(__file__), 'output_test_01.html')
with codecs.open(file_path_3, "w", "utf-8") as file:
    file.write(html_content)


###########################################################################################################
################################# 출력된 HTML 파일을 뉴스레터로 보내는 코딩##################################
###########################################################################################################

import smtplib
    # 파이썬에서 이메일을 보내기 위해서 불러와야 한다. 
import email
    # 이메일 메시지 작성, 헤더와 본문의 설정, MIME(Multipurpose Internet Mail Extensions)형식으로 변환. (MIME형식으로 변환해야 전송이 가능)
from email.mime.multipart import MIMEMultipart
    # 이메일 메시지(이하 '이메일') 각 부분(송신자/수신자/제목/별첨/본문 등)을 객체로 생성후 각 부분을 변수로 할당해 정보를 지정할 수 있도록 만듦.
from email.mime.text import MIMEText
    # text 데이터를 MIME형식으로 변환하는데 사용. MIMEText클래스는 이메일 본문 내용 작성시 사용
from email.mime.base import MIMEBase
    # 이메일 메시지에 첨부파일(이미지/동영상/음성파일 등 다양한 형식)을 추가하기 위해 사용.
from email import encoders
    # 이진(binary) 데이터를 MIME 형식으로 인코딩하기 위한 함수를 제공.
from email.mime.image import MIMEImage
    # 이메일 본문(body)에 이미지를 붙이기 위해서 사용
from PIL import Image
    # Pillow(PIL)은 html을 이미지로 이메일 본문에 보여주기 위해서 설치한 것임.
import imghdr
    # imghdr 모듈은 이미지 파일의 형식을 확인하는 표준 라이브러리. 
import base64
import io


SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
SMTP_USER = '??????????????????????????????'  # 각 개인의 이메일 아이디
SMTP_PASSWORD = '??????????????????????????'  # 각 개인이 설정한 SMTP 비밀번호 입력

to_users = ['kjy153@naver.com', 'kjy153@gmail.com']
target_addr = ','.join(to_users)

subject = '제목: test news email by kimjongwoo from NAVER'
contents = '김종우가 송신합니다.변경되지 않는 내용을 여기에 명시하면 됩니다. 변경될 내용은 기사요약입니다.'

# 이메일 송신인 / 수신인 / 제목 / 본문내용 생성
msg = MIMEMultipart('related')  # mixed가 아닌 related로 설정해도 별첨화일을 송부할 수 있다.

msg['From'] = SMTP_USER   #MIMEMultipart객체의 송신인 필드에 값을 넣어준 것임
msg['To'] = target_addr   #MIMEMultipart객체의 수신인 필드에 값을 넣어준 것임
msg['Subject'] = subject  #MIMEMultipart객체의 제목 필드에 값을 넣어준 것임


# 이메일에 첨부할 html 파일 추가
file_name_4 = os.path.basename(file_path_3) 
    #'output_test_01.html'파일명과 경로를 ' file_path_3 '에 이미 할당해 놓았기 때문에  os.path.basename()메서드로 연결시켜서 'output_test_01.html'파일명을 2번 써 줄 필요가 없도록 코딩(실수가능성차단)  
file_path_4 = os.path.join(os.path.dirname(__file__), file_name_4)
with open(file_path_4, 'r', encoding='utf-8') as f:
    file_data = f.read()

email_file = MIMEBase('text', 'html', charset='UTF-8')
email_file.set_payload(file_data.encode('UTF-8'))
encoders.encode_base64(email_file)
email_file.add_header('Content-Disposition', 'attachment', filename=file_name_4)  
msg.attach(email_file)

# 이미지를 이메일 본문안에 Inline으로 붙여넣기 
file_name_pic_1 = 'img_newsletter.jpg'      ######### <===== 사진파일이름 저장
file_path_5 = os.path.join(os.path.dirname(__file__), file_name_pic_1)
with open(file_path_5, 'rb') as file:
    img_data = file.read()
    img = Image.open(io.BytesIO(img_data))  
img_type = imghdr.what(None, img_data)
image = MIMEImage(img_data, _subtype=img_type)
image.add_header('Content-ID', '<image1>')
# msg.attach(image) # 바로 옆 코드를 활성화시키면 image화일이 첨부된다. 이메일 본문에만 이미지 붙이기 위해 비활성화시켜 놓았슴.
encoded_string = base64.b64encode(open(file_path_5, 'rb').read()).decode()
html = f"""\
<html>
  <body>
    <p>{contents}</p>
    <p> 여기 바로 아래에 이미지가 붙습니다. 이미지 설명을 여기에 하세요. </p>
    <p><img alt="ImageLoadingErr" width="805" height="502" style="border:none;"  # img alt="ImageLoadingErr"는 이미지가 출력되지 않을 경우 보여질 에러 메시지
    src="data:image/jpeg;base64,{encoded_string}" /></p>
  </body>
</html>
"""
html = MIMEText(html, 'html')   # 위에서 html 문자열로 만든 이미지를 변수 html에 할당
msg.attach(html)                # 이미지가 할당된 변수 html을 이메일 메시지 본문에 붙여주는 코딩이다. 

# 기사요약(output_test_01.html) 파일의 내용을 이메일 메시지 본문에 붙이는 코딩이다.
html = MIMEText(file_data, 'html')
msg.attach(html)


smtp = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
smtp.login(SMTP_USER, SMTP_PASSWORD)
smtp.sendmail(SMTP_USER, to_users, msg.as_string())
smtp.close()

###############################################################################
########################### Telegram으로 전송###################################
###############################################################################



import telegram
from telegram import InputFile
import asyncio
import requests
from datetime import datetime


# 텔레그램 봇 생성
#token = '????????????????????????????????????'         # 텔레그램 토큰 비밀번호를 bot에서 생성하여 붙임
#bot = telegram.Bot(token)                              # 코딩 두 줄을 아래와 같이 한줄로 코딩해도 된다.
bot = telegram.Bot(token='???????????????????')         # 텔레그램 토큰 비밀번호를 bot에서 생성하여 붙임

# 비동기 함수(async def main()) 구문으로 작성
async def main():   
    current_time = datetime.now()    
    await bot.send_message(chat_id='?????????????????', text=f'테스트전송시각 for NewsLetter_전송시각은 {current_time}')

    # 전송할 이미지 파일과 그 경로 및 전송 실행
    file_name_6 = os.path.basename(file_path_5) 
    file_path_6 = os.path.join(os.path.dirname(__file__), file_name_6)
    with open(file_path_6, 'rb') as f:
        await bot.send_photo(chat_id='6236837190', photo=InputFile(f, filename='file_name_6'))
        
    # 전송할 html 파일과 그 경로 및 전송 실행
    file_name_7 = os.path.basename(file_path_3) 
    file_path_7 = os.path.join(os.path.dirname(__file__), file_name_7)
    with open(file_path_7, 'rb') as f:
        await bot.send_document(chat_id='6236837190', document=InputFile(f, filename=f'news letter_[{current_time}].html'))

if __name__ == '__main__':
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
