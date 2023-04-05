import re
import pandas as pd

df = pd.read_csv('./total.csv')

for i in range(len(df)):
    # <,( 로 감싸진 경우도 작동하고 언론사도 검사하는 코드 필요
    # 괄호안에 있는 작성자, 언론사를 제대로 제거하지 못하면 개행 활용 제거 부분에서 데이터 손실
    regex_parenthesis  = '\▶(.*)|[^\u0000-\u007F\uAC00-\uD7A3]|([(\[])(.*?)([)\]])|[\w\.-]+@[\w\.-]+\.[\w\.]+|([^\s]+ 기자\s*=)'

    test = re.sub(regex_parenthesis, '', df['content'][i])
    test = test.split('\n')
    new = []
    for text in test:
        if df['writer'][i] not in text:
            if df['news_agency'][i] not in text:
                # tmp = re.sub(regex_email, '', text)
                new.append(text)
        
    new = ' '.join(str(s) for s in new)
    new = ' '.join(str(s) for s in new.split())

    print(new)
    print(df['url'][i])