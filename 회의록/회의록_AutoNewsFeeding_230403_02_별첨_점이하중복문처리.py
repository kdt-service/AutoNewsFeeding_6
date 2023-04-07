import re
import pandas as pd

df = pd.read_csv("news.csv")

total_content = []
for i in range(len(df)):
    content = df['content'][i]
    content = content.split()
    content = ' '.join(content)

    # 마침표를 마침표 및 개행으로 변경하기 위해 사용
    regex_sentence = '\.'

    # 마침표를 마침표 및 개행으로 바꾼다.
    test = re.sub(regex_sentence, '.\n', content)

    # 개행으로 문자를 나눈어 언론사 및 작성자가 들어간 내용을 삭제한다.
    test = test.split('\n')
    if len(test) > 2:
        print(test[-1])
        total_content.append(test[-1])

total = pd.DataFrame(total_content)
total.to_csv('end.csv')