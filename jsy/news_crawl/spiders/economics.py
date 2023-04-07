import scrapy
import re
import traceback
from datetime import datetime, timedelta

class EconomicsSpider(scrapy.Spider):
    name = "economics"

    CATEGORIES = {
        ('경제', 'economic') : {
            '금융': 'finance',
            '기업산업': 'industry',
            '취업직장인': 'employ',
            '경제일반': 'others',
            '자동차': 'autos',
            '주식': 'stock',
            '시황분석': 'stock/market',
            '공시': 'stock/publicnotice',
            '해외증시': 'stock/world',
            '채권선물': 'stock/bondsfutures',
            '외환': 'stock/fx',
            '주식일반': 'stock/others',
            '부동산': 'estate',
            '생활경제': 'consumer',
            '국제경제': 'world'
        }
    }

    URL_FORMAT = 'https://news.daum.net/breakingnews/{}/{}?page={}&regDate={}'
    def start_requests(self):
        start_date = datetime(2023, 2, 1)
        end_date = datetime(2023, 2, 3)
        dates = [start_date]
        while True:
            start_date = start_date + timedelta(days=1)
            dates.append(start_date)
            if start_date == end_date:
                break

        for main_ctg in self.CATEGORIES:
            main_name, main_id = main_ctg
            for sub_name, sub_id in self.CATEGORIES[main_ctg].items():
                for date in dates:
                    target_url = self.URL_FORMAT.format(main_id, sub_id, 1, date.strftime('%Y%m%d'))

                    yield scrapy.Request(url=target_url, callback=self.parse_url, meta={
                        'page': 1, 'urls': [], 'main_category': main_name, 'sub_category': sub_name})

    def parse_url(self, response):
        # 페이지 체크
        urls = response.css('a.link_thumb::attr(href)').getall()
        if response.meta.pop('urls') == urls:
            return
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={**response.meta})

        # 다음페이지
        page = response.meta.pop('page')
        target_url = re.sub('page\=\d+', f'page={page+1}', response.url)
        yield scrapy.Request(url=target_url, callback=self.parse_url, meta={**response.meta, 'page': page+1, 'urls':urls})

    def parse(self, response):
        try:
            title = response.css('.tit_view::text').get().strip()
            content = response.css('.article_view')[0].xpath('string(.)').extract()[0].strip()
            infos = response.css('.info_view .txt_info')
            if len(infos) == 1:
                writer = ''
                writed_at = infos[0].css('.num_date::text').get()
            else:
                writer = response.css('.info_view .txt_info::text').get()
                writed_at = infos[1].css('.num_date::text').get()
            
            news_id = response.url.split('/')[-1]
            with open('./news_contents/'+news_id+'.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            
            datas = [news_id, response.meta.pop('main_category'), response.meta.pop('sub_category'), title, writer, writed_at]

            with open('./metadata.tsv', 'a', encoding='utf-8') as f:
                f.write('\t'.join(map(str,datas)) + '\n')

        except:
            traceback.print_exc()
            with open('error_urls', 'a') as f:
                f.write(response.url+'\n')