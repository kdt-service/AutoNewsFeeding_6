import os
import pandas as pd
from konlpy.tag import Mecab
from gensim import corpora, models
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer    


class Summary:
    
    def __init__(self, num_sentences=3):
        self.num_sentences = num_sentences
        self.korean_stop_words = set()
    
    def stopword(self, file_path='stop_word.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.korean_stop_words = set(line.strip() for line in f)

    def preprocess(self, text): 
        mecab = Mecab(dicpath=r'/tmp/mecab-ko-dic-2.1.1-20180720')
        tokens = [word for word in mecab.pos(text) if word[1][0] in ['N', 'V']]
        tokens = [word for word, pos in tokens if word not in self.korean_stop_words]
        return tokens

    def remove_sources(self, text, sources=['출처=IT동아', '사진=트위터 @DylanXitton','▶△▶️◀️▷ⓒ■◆●©️…※↑↓▲☞ⓒ⅔▼�','([(\[])(.*?)([)\]])|【(.*?)】','<(.*?)>','제보는 카톡 okjebo','  ']):
        for source in sources:
            text = text.replace(source, '')
        return text
    
    def sentence_similarity(self, sentence1, sentence2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def lda_summarize_v2(self, text, num_topics=3, num_words=10, similarity_threshold=0.4):
        text = self.remove_sources(text)
        sentences = sent_tokenize(text)
        tokenized_sentences = [self.preprocess(sentence) for sentence in sentences]
        dictionary = corpora.Dictionary(tokenized_sentences)
        corpus = [dictionary.doc2bow(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
        lda_model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        topic_keywords = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        topic_sentences = []

        for topic, keywords in topic_keywords:
            topic_words = [word[0] for word in keywords]
            for sentence in sentences:
                tokenized_sentence = self.preprocess(sentence)
                if any(word in tokenized_sentence for word in topic_words):
                    topic_sentences.append(sentence)

        ranked_sentences = sorted([(sentence, lda_model[corpus[sentences.index(sentence)]][0][1]) for sentence in topic_sentences], key=lambda x: x[1], reverse=True)

        summary = []
        for sentence, score in ranked_sentences:
            if len(summary) >= self.num_sentences:
                break
        
            for chosen_sentence, i in summary:
                if self.sentence_similarity(sentence, chosen_sentence) > similarity_threshold:
                    break
        
            else:
                summary.append((sentence, score))

        summary_sentences = [sentence[0] for sentence in summary]
        return summary_sentences


    def generate_html(self, df):
        html_content = '''<div lang="en" style="-webkit-text-size-adjust: 100%;-ms-text-size-adjust: 100%;padding: 20px 0px;margin: 0 auto;"><div style="height:0px;max-height:0px;border-width:0px;border: 0px;border-color:initial;border-image:initial;visibility:hidden;line-height:0px;font-size:0px;overflow:hidden;display:none;"></div><table><tbody><tr><td></td></tr></tbody></table><table style="width:100%;" border="0"><tbody><tr><td align="center"><!--[if mso]>          <table align="center" style="width:630px;background:#ffffff;">          <tr><td>            <div>          <![endif]--><!--[if !mso]><!-- --><div style="width:100%;max-width:630px;background:#ffffff;margin: 0px auto;"><!--&lt;![endif]--><table width="100%" cellpadding="0" cellspacing="0" style="border:0;"><tbody><tr><td style="padding:0 0;text-align:left;margin:0px;line-height:1.7;word-break:break-word;font-size:12px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#747579;"><table border="0" cellpadding="0" cellspacing="0" style="width: 100%"><tbody><tr></tr></tbody></table></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="text-align:center;font-size:0;box-sizing:border-box;padding:5px 15px 5px 15px;"><img src="https://img.stibee.com/107_1619587979.png" alt="" style="display:inline;vertical-align:bottom;text-align:center;max-width:100% !important;height:auto;border:0;" width="600" loading="lazy"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="text-align:center;font-size:0;box-sizing:border-box;padding:15px 15px 0 15px;"><img src="https://img.stibee.com/107_1619587988.jpg" alt="" style="display:inline;vertical-align:bottom;text-align:center;max-width:100% !important;height:auto;border:0;" width="600" loading="lazy"></td></tr><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:20px 15px 15px 15px;"><div>취업 준비를 하다보면, '이 많은 뉴스를 어떻게 다 읽지?' 하고 막막할 때가 있습니다. 짧은 시간 안에 내가 필요한 뉴스만 읽을 수 있다면? 그 이상을 실현시켜주는 뉴스레터가 왔습니다. 내가 원하는 기업의 기사만 골라서 정리해주는 '똑똑한 뉴스레터'와 함께 똑똑하게 취준하세요!</div></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;"><tbody><tr><td style="height:15px;" colspan="3"></td></tr><tr><td style="width:15px;"></td><td style="height:15px;background: none;padding: 0px;border-top-width:1px;border-top-style:dashed;border-top-color:#999999;margin:0 0;"><!-- &lt;div style='background: none;padding: 0px;border-top-width:1px;border-top-style:dashed;border-top-color:#999999;margin:0 0;'&gt;&lt;/div&gt; --></td><td style="width:15px;"></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 15px 25px 15px;"><div style="text-align: center;"><span style="font-size: 20px; font-weight: bold;">💌</span><br><span style="font-size: 20px;"><span style="font-weight: bold;">취준레터가 도착했습니다!</span></span></div></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 15px 15px 15px;"><div><span style="font-weight: bold; font-size: 18px;">구독자님이 요청한 기업과 관련한 기사를 요약한 내용입니다.</span><br><span style="color: #959595;"></span></div><div><br></div>'''
        
        for index, row in df.iterrows():
            content = row['content']
            summary = self.lda_summarize_v2(content, num_topics=3, num_words=10, similarity_threshold=0.4)
            url = row['url']

            html_content += f'<div><span style="text-decoration: underline;">{index+1}. {row["title"]}</span></div><div>- '
            for i in range(min(self.num_sentences, len(summary))):
                html_content += f"{summary[i]} "
            html_content += f'<div>- <a href="{url}" target="_blank" style="text-decoration: underline; color: #ff5252;" rel="noreferrer noopener">{url}</a></div>'
            html_content += "</div><div><br></div>"
        
        html_content += '''<table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table style="width: 100%;background:#F1F3F5;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:25px 15px 25px 15px;"><div style="text-align: center;"><span style="font-weight: bold;">오늘의 뉴스레터는 여기까지</span><br>구독자님, 취준레터를 읽어주셔서 감사해요. 우리 다음주에도 건강히 또 만나요!</div><div style="text-align: center;"><br><span style="font-weight: bold;">오늘의 뉴스레터는 어땠나요?</span><br><span style="color: #ff5252; text-decoration: underline;">좋았어요! 🤗</span>ㅣ <span style="color: #ff5252; text-decoration: underline;">음, 잘 모르겠어요 🥺</span></div></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td width="100%" style="clear:both;;"><table border="0" cellpadding="0" cellspacing="0" style="width:100%;"><tbody><tr><td style="padding:25px 15px 15px 15px;border:0;width:100%;text-align:center;"><table border="0" cellpadding="0" cellspacing="0" style="color:#ff5252;mso-padding-alt:0px;background:#ffffff;border-radius:500px;border:2px solid #ff5252;mso-line-height-rule:exactly;box-sizing:border-box;text-decoration:none;outline:0px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;text-align:center;display:inline-table;"><tbody><tr><td valign="top" style="font-size:0;background:none;border-radius:500px;padding:19px 20px;text-align:center;"><a href="#" style="font-size:16px;display:inline-block;color:#ff5252;text-decoration:none;outline:0px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, malgun gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, meiryo, ms gothic, sans-serif;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;text-align:center;line-:1;box-sizing:border-box;mso-line--rule:exactly;" target="_blank" rel="noreferrer noopener">뉴스레터 구독하기</a></td></tr></tbody></table><table align="center" border="0" cellpadding="0" cellspacing="0"><tbody><tr></tr></tbody></table></td></tr></tbody></table></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;"><tbody><tr><td style="text-align:center;margin:0px;;line-height:1.7;word-break:break-word;font-size:12px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#747579;border:0;"><table border="0" cellpadding="0" cellspacing="0" style="width: 100%"><tbody><tr><td style="padding:15px 15px 15px 15px;text-align:center;"><div><span style="color: #959595;"><span style="font-weight: bold;">취준레터</span> by newsfeeding_team6</span><br><a href="https://github.com/kdt-service/AutoNewsFeeding_6" target="_blank" style="color: #747579;" rel="noreferrer noopener">https://github.com/kdt-service/AutoNewsFeeding_6</a><br><span style="color: #959595;"><a style="text-decoration: underline; color: #959595;" href="https://stibee.com/error/preview" target="_blank" rel="noreferrer noopener"><span style="font-weight: normal; font-style: normal;">수신거부</span></a>&nbsp;<a style="text-decoration: underline; color: #959595;" href="https://stibee.com/error/preview" target="_blank" rel="noreferrer noopener"><span style="font-weight: normal; font-style: normal;">Unsubscribe</span></a></span></div></td></tr></tbody></table></td></tr></tbody></table></div><!--[if mso]>          </div></td></tr>          </table>          <![endif]--></td></tr></tbody></table><table align="center" border="0" cellpadding="0" cellspacing="0" width="100%"><tbody><tr></tr></tbody></table><div style="padding: 10pt 0cm 10pt 0cm;"><p align="center" style="text-align: center;"><span lang="EN-US"></span></p></div></div>'''

        count = len(os.listdir('./news'))   # news폴더 내의 파일목록 갯수

        file_path = f'news/output_test_{count}.html'
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(html_content)