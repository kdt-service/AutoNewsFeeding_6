import re
import pandas as pd

df = pd.read_csv('./total.csv')

total_content = []
for i in range(len(df)):
    # 괄호안에 있는 작성자, 언론사를 제대로 제거하지 못하면 개행 활용 제거 부분에서 데이터 손실
    content = df['content'][i]

    # regex_basics로 큰따옴표, 작은 따옴표 지워지는 것 방지
    content.replace('“','"').replace('”','"')
    content.replace("‘","'").replace("’","'")

    # 한글 키보드로 입력 불가능한 문자 제거, 한자키가 필요한 문자들
    regex_basics = '[^\u0000-\u007F\uAC00-\uD7A3]'
    # 이메일 제거(.을 지우기 위해)
    regex_email = '\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'
    # URL 제거(.을 지우기 위해)
    regex_url = '(https?://)?([\da-z\.-]+)\.([a-z\.]{2,6})([/\w\.-]*)*/?'
    # 괄호 안에 있는 문자 작성자, 언론사 제거
    regex_parenthesis_1 = '[\(\{\[\<].*%s.*?[\)\}\]\>]'%(df['writer'][i])
    regex_parenthesis_2 = '[\(\{\[\<].*%s.*?[\)\}\]\>]'%(df['news_agency'][i])
    # 중요 내용 지워지는 위험 방지(기자명, 언론사명이 지워지지 않는 곳)
    regex_exception = '\▶️(.*)|([(\[])(.*?)([)\]])|[\w\.-]+@[\w\.-]+\.[\w\.]+|([^\s]+ 기자\s*=)'
    # 마침표를 마침표 및 개행으로 변경하기 위해 사용
    regex_sentence = '\.'

    # 한자키 사용 특수 문자의 의미를 잃지 않기 위해서 띄어쓰기 추가
    test = re.sub(regex_basics, ' ', content)
    test = re.sub(regex_email, '', content)
    test = re.sub(regex_url, '', content)
    test = re.sub(regex_parenthesis_1, '', test)
    test = re.sub(regex_parenthesis_2, '', test)
    test = re.sub(regex_exception, '', test)
    # 마침표를 마침표 및 개행으로 바꾼다.
    test = re.sub(regex_sentence, '.\n', test)

    # 개행으로 문자를 나눈어 언론사 및 작성자가 들어간 내용을 삭제한다.
    test = test.split('\n')
    new = []
    for text in test:
        if df['writer'][i] not in text:
            if df['news_agency'][i] not in text:
                new.append(text)
    
    # 새로운 본문 생성
    new = ' '.join(str(s) for s in new)
    new = ' '.join(str(s) for s in new.split())

    total_content.append(new)