# 뉴스 파일의 전체 로직

## 1. 크롤링한 뉴스기사 데이터를 불러온다.
## 2. 뉴스기사 데이터를 전처리
## 3. 뉴스기사 키워드 추출, 요약
## 4. 텔레그램 혹은 이메일을 통해 사용자에게 발송


# 파일 불러오기 방법 1, 2
# pd.read_csv('/home/ubuntu/workspace/news.csv')

# files = os.listdir('./news')
# '1'

# last_news_number = int(open('./last_csv_file_number').read())

# # for f in files[last_news_number:]:
#     # df에 추가
#     # if int(f.split('.')[0].split('_')[1]) > last_news_number:
#         # df에 추가
# # else:
# #     # last_csv_file_number에 추가
# #     int(f.split('.')[0].split('_')[1])
        


if __name__ == '__main__':
    print('뉴스 기사 요약 시작 !')

