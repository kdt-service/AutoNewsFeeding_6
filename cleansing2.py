import re
import pandas as pd

df = pd.read_csv('./total.csv')

total_content = []
for i in range(len(df)):
    # 괄호안에 있는 작성자, 언론사를 제대로 제거하지 못하면 개행 활용 제거 부분에서 데이터 손실
    content = df['content'][i]

    content.replace('“','"').replace('”','"')
    content.replace("‘","'").replace("’","'")

    regex_basics = '[^\u0000-\u007F\uAC00-\uD7A3]'
    regex_email = '\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'
    regex_url = '(https?://)?([\da-z\.-]+)\.([a-z\.]{2,6})([/\w\.-]*)*/?'
    regex_parenthesis_1 = '[\(\{\[\<].*%s.*?[\)\}\]\>]'%(df['writer'][i])
    regex_parenthesis_2 = '[\(\{\[\<].*%s.*?[\)\}\]\>]'%(df['news_agency'][i])
    regex_exception = '\▶️(.*)|([(\[])(.*?)([)\]])|[\w\.-]+@[\w\.-]+\.[\w\.]+|([^\s]+ 기자\s*=)'
    regex_sentence = '\.'

    test = re.sub(regex_basics, ' ', content)
    test = re.sub(regex_email, '', content)
    test = re.sub(regex_url, '', content)
    test = re.sub(regex_parenthesis_1, '', test)
    test = re.sub(regex_parenthesis_2, '', test)
    test = re.sub(regex_exception, '\n', test)
    test = re.sub(regex_sentence, '.\n', test)
    test = test.split('\n')
    new = []
    for text in test:
        if df['writer'][i] not in text:
            if df['news_agency'][i] not in text:
                new.append(text)
        
    new = ' '.join(str(s) for s in new)
    new = ' '.join(str(s) for s in new.split())

    total_content.append(new)