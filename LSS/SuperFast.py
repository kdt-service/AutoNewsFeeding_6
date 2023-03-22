import os
import sys
import grequests
import requests
import pandas as pd
import multiprocessing
import concurrent.futures
from bs4 import BeautifulSoup

sys.setrecursionlimit(10000)


class DaumNewsCrawler:
    CATEGORIES = {
        'economic': ['경제', {
            'finance': '금융',
            'industry': '기업산업',
            'employ': '취업직장인',
            'others': '경제일반',
            'autos': '자동차',
            'stock': '주식',
            'stock/market': '시황분석',
            'stock/publicnotice': '공시',
            'stock/world': '해외증시',
            'stock/bondsfutures': '채권선물',
            'stock/fx': '외환',
            'stock/others': '주식일반',
            'estate': '부동산',
            'consumer': '생활경제',
            'world': '국제경제'
        }],
        'digital': ['IT', {
            'internet': '인터넷',
            'science': '과학',
            'game': '게임',
            'it': '휴대폰통신',
            'device': 'IT기기',
            'mobile': '통신_모바일',
            'software': '소프트웨어',
            'others': 'Tech일반'
        }],
        'culture': ['문화', {
            'health': '건강',
            'life': '생활정보',
            'art': '공연/전시',
            'book': '책',
            'leisure': '여행레져',
            'others': '문화생활일반',
            'weather': '날씨',
            'fashion': '뷰티/패션',
            'home': '가정/육아',
            'food': '음식/맛집',
            'religion': '종교'
        }]
    }

    def __init__(self, main_category, sub_category, date):
        self.base_url = f"https://news.daum.net/breakingnews/{main_category}/{sub_category}?regDate={date}&page="
        self.last_page = 0
        self.page_list = []
        self.news_list = []
        self.index = 0
        self.main_category = main_category
        self.sub_category = sub_category
        self.date = date
        self.data_df = pd.DataFrame(columns=[
                                    'platform', 'main_category', 'sub_category',
                                    'title', 'content', 'writer', 'writed_at', 'news_agency'])
        self.params = {}

    def find_last_page(self):
        last_url = self.base_url + "9999"
        response = requests.get(last_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tmp_page = soup.select_one('.inner_paging em').text
        self.last_page = int(tmp_page.replace('현재 페이지', '').strip())

    def search(self):
        for date in range(1, self.last_page+1):
            self.page_list.append(self.base_url + str(date))

        reqs = (grequests.get(link) for link in self.page_list)
        resp = grequests.map(reqs)

        for r in resp:
            soup = BeautifulSoup(r.text, 'lxml')
            news_list = soup.select('.list_allnews')
            news_list = news_list[0].select('.tit_thumb')

            for news in news_list:
                link = news.find('a')
                self.news_list.append(link.attrs['href'])

    def crawling(self):
        reqs = (grequests.get(link) for link in self.news_list)
        resp = grequests.map(reqs)

        for r in resp:
            data = []
            soup = BeautifulSoup(r.text, 'lxml')
            title = soup.select_one('.head_view > .tit_view').text
            writer = soup.select('.head_view > .info_view > .txt_info')[0].text
            writed_at = soup.select_one(
                '.head_view > .info_view .num_date').text
            content = soup.select_one('.news_view > .article_view')
            news_agency = soup.select_one('#kakaoServiceLogo').text
            data += '다음뉴스', self.CATEGORIES[self.main_category][0], \
                self.CATEGORIES[self.main_category][1][self.sub_category], \
                title, content, writer, writed_at, news_agency
            self.data_df.loc[self.index] = data
            self.index += 1

    def save(self):
        self.data_df.to_csv(
            f'./{self.CATEGORIES[self.main_category][0]}_{self.CATEGORIES[self.main_category][1][self.sub_category]}_{self.date}.csv')

    def start(self):
        self.find_last_page()
        self.search()
        self.crawling()
        self.save()
        return None


def get_date_list(start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    date_list = pd.date_range(start=start, end=end, freq='D')
    date_list = date_list.strftime('%Y%m%d')
    return date_list


def combine_all_csv(name):
    csv_list = []
    files = os.listdir(os.getcwd())

    for file in files:
        name, ext = os.path.splitext(file)
        if ext == '.csv':
            csv_list.append(file)

    total_csv = pd.DataFrame(columns=[
        'platform', 'main_category', 'sub_category',
        'title', 'content', 'writer', 'writed_at', 'news_agency'
    ])
    for csv in csv_list:
        df = pd.read_csv(csv)
        total_csv = pd.concat([total_csv, df], ignore_index=True)
    
    total_csv.to_csv(f'./{name}.csv')

    for csv in csv_list:
        os.remove(csv)


if __name__ == '__main__':
    date_list = get_date_list('20230201', '20230316')

    sample = DaumNewsCrawler('economic', 'finance', '20230201')

    class_list = []
    total_df = sample.data_df

    for date in date_list:
        for category in sample.CATEGORIES['digital'][1].keys():
            class_list.append(DaumNewsCrawler('digital', category, date))

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()*2)

    procs = []

    for dnc in class_list:
        procs.append(pool.submit(dnc.start))

    for p in concurrent.futures.as_completed(procs):
        pass

    combine_all_csv('total.csv')
