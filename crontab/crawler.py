# 크롤러 파일의 전체 로직

## 1. 뉴스 요약에 사용할 데이터를 크롤링 (BeautifulSoup or Scrapy)
### 1-1 . 뉴스 기사를 크롤링 할 시 전체 뉴스기사를 크롤링하는 것이 아니라, 
###        만약에 수집을 카테고리별로 한다고 하면, 카테고리별로 마지막 뉴스의 news_id나 url을 파일로 저장해놓고,
###        현재 크롤링하는 뉴스기사의 url이 위에서 저장해놓은 url과 일치하면, 거기서 크롤링을 멈추면 된다.

## 2. 크롤링한 데이터를 정리 (딕셔너리, 데이터프레임)
### 2-1. 데이터를 저장하기 전, 클렌징을 통해 데이터를 우선적으로 정리한다.

## 3. 데이터를 저장 
### 3-1. 파일의 경로는 뒤에서 사용할 뉴스기사 요약 .py 파일에서 읽어올 경로와 맞춰주어야한다.

# 데이터 저장 시 방법 1, 2
# df.to_csv('/home/ubuntu/workspace/news.csv')

# length = len(os.listdir('./news')) # => 0 -> 1
# f'news_{length}.csv'

import os
if __name__ == '__main__':
    ## 실행되는 경로에 상관없이 프로그램이 특정 경로를 가지게 하기
    ## crontab으로 실행하여 현재 경로 확인
    print(os.path.abspath('.')) # 현재 경로를 출력
    os.chdir('../../../test') # 현재 경로를 바꿈
    
    # 상대경로로 파일생성 확인
    with open('./testfile.txt', 'w') as f:
        f.write('hello I\'m here!')

