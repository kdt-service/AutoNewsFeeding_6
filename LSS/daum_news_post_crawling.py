from bs4 import BeautifulSoup
import requests
import pandas as pd
import multiprocessing
import concurrent.futures


def get_record(i, url):
    print(i, url)
    post = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.select_one('.head_view > .tit_view')
    post.append(title.text)

    writer = soup.select('.head_view > .info_view > .txt_info')[0]
    post.append(writer.text)

    date = soup.select_one('.head_view > .info_view .num_date')
    post.append(date.text)

    content = soup.select_one('.news_view > .article_view')
    post.append(content)

    return i, post


if __name__ == '__main__':
    link_df = pd.read_csv('./news_list.csv')
    post = {
        'title': [],
        'writer': [],
        'date': [],
        'content': []
    }
    post_df = pd.DataFrame(post)

    print('start df work')
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()*2)
    procs = []

    for i, url in enumerate(link_df['link']):
        procs.append(pool.submit(get_record, i, url))

    for p in concurrent.futures.as_completed(procs):
        i, post = p.result()
        post_df.loc[i] = post

    post_df.to_csv('raw_data.csv')
