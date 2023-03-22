from bs4 import BeautifulSoup
import requests
import pandas as pd
import multiprocessing
import concurrent.futures


def crawling(date):
    url = "https://news.daum.net/breakingnews/economic"
    page = 1
    page_max = 0
    link_list = []

    while True:
        params = {
            "regDate": str(date),
            "page": str(page)
        }
        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = soup.select('.list_allnews')
        news_list = news_list[0].select('.tit_thumb')

        page_check = soup.select('a.btn_page.btn_next')

        if page_check == [] and page_max == 0:
            try:
                page_max = int(soup.select('.num_page')[-1].text)
            except:
                page_max = int(soup.select('.num_page')
                               [-1].text.strip().replace('현재 페이지', ''))
        elif page_max != 0:
            if page_max < page:
                break

        for news in news_list:
            link = news.find('a')
            print(link)
            link_list.append(link.attrs['href'])
        page += 1
    return link_list


def get_date_list(start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    date_list = pd.date_range(start=start, end=end, freq='D')
    date_list = date_list.strftime('%Y%m%d')

    return date_list


if __name__ == '__main__':

    news_list = []
    procs = []

    date_list = get_date_list('20230201', '20230316')

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()*2)

    for date in date_list:
        procs.append(pool.submit(crawling, date))

    for p in concurrent.futures.as_completed(procs):
        news_list += p.result()

    print(len(news_list))

    with open(f'./news_list.csv', 'w') as f:
        field = 'link' + '\n'
        f.write(field)
        record = '\n'.join([record for record in news_list])
        f.write(record)
