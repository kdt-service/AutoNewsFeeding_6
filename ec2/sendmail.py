import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders  
from email.mime.image import MIMEImage
from PIL import Image
import imghdr
import base64
import io
import os


SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
SMTP_USER = '?????????????????????????????????????'  # 각 개인의 이메일 아이디
SMTP_PASSWORD = '?????????????????????????????????'  # 각 개인이 설정한 SMTP 비밀번호 입력

to_users = ['kjy153@naver.com',  'kjy153@gmail.com'] # 수신인 주소 기입하는 곳
target_addr = ','.join(to_users)

subject = '제목: test news email by kimjongwoo from NAVER'
contents = '변경되지 않는 내용을 여기에 명시하면 됩니다. 변경될 내용은 기사요약입니다.'

# 이메일 송신인 / 수신인 / 제목 / 본문내용 생성
msg = MIMEMultipart('related')  # mixed가 아닌 related로 설정해도 별첨화일을 송부할 수 있다.

msg['From'] = SMTP_USER   #MIMEMultipart객체의 송신인 필드에 값을 넣어준 것임
msg['To'] = target_addr   #MIMEMultipart객체의 수신인 필드에 값을 넣어준 것임
msg['Subject'] = subject  #MIMEMultipart객체의 제목 필드에 값을 넣어준 것임

#text = MIMEText(contents)   # 충돌은 아니고, 겹치는 것을 피하기 위해 비활성화시켰슴
#msg.attach(text)            # 충돌은 아니고, 겹치는 것을 피하기 위해 비활성화시켰슴
    # 아래 2개의 코드 html = MIMEText(html, 'html') msg.attach(html)와 겹치는 것을 
    # 피하기 위해 위 2개의 코드를 비활성화시킴


# 이메일에 첨부할 html 파일 추가
file_name_4 = os.path.basename(file_path_3) 
    #'output_test_01.html'파일명과 경로를 ' file_path_3 '에 이미 할당해 놓았기 때문에 
    # os.path.basename()메서드로 연결시켜서 'output_test_01.html'파일명을 2번 써 줄 필요가 없도록 코딩(실수가능성차단)  
file_path_4 = os.path.join(os.path.dirname(__file__), file_name_4)
with open(file_path_4, 'r', encoding='utf-8') as f:
    file_data = f.read()

email_file = MIMEBase('text', 'html', charset='UTF-8')
email_file.set_payload(file_data.encode('UTF-8'))
encoders.encode_base64(email_file)
email_file.add_header('Content-Disposition', 'attachment', filename=file_name_4)  
msg.attach(email_file)

# 이메일 본문에 붙여넣을 html 코딩은 아래와 같다. 그림이 먼저 나오게 하기 위해서 아래 코드를
# 비활성시키고, 복사해서 맨 아래에 붙여 넣는다. 그러면 그림(이미지)이 먼저 나오고 그 다음에 기사요약이 나올 것이다.
# html = MIMEText(file_data, 'html')
# msg.attach(html)

file_name_pic_1 = 'img_newsletter.jpg'  # 사진파일이름 저장
file_path_5 = os.path.join(os.path.dirname(__file__), file_name_pic_1)
with open(file_path_5, 'rb') as file:
    img_data = file.read()
    img = Image.open(io.BytesIO(img_data))  
img_type = imghdr.what(None, img_data)
# Create a MIMEImage object and set the Content-ID header
image = MIMEImage(img_data, _subtype=img_type)
image.add_header('Content-ID', '<image1>')
#msg.attach(image) # 이 코드를 활성화시키면 image화일이 첨부된다.


encoded_string = base64.b64encode(open(file_path_5, 'rb').read()).decode()

#html = '''<img alt="ImageLoadingErr" width="805" height="502" style="border:none;" src="data:image/jpeg;base64,{logo}" />'''.format(logo=encoded_string)
    # 위 코딩을 비활성화시키고 가시성을 높이기 위해 아래 코딩으로 변경하였다. 동일하다.
html = f"""\
<html>
  <body>
    <p>{contents}</p>
    <p>여기는 이미지가 붙을 자리입니다. 여기 아래에 이미지가 붙어야 합니다.</p>
    <p><img alt="ImageLoadingErr" width="805" height="502" style="border:none;" 
    src="data:image/jpeg;base64,{encoded_string}" /></p>
  </body>
</html>
"""
html = MIMEText(html, 'html')
msg.attach(html)

# 기사요약 으로 출력한 output_test_01.html파일의 내용을 기사본문에 붙이는 코딩이다.
html = MIMEText(file_data, 'html')
msg.attach(html)


smtp = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
smtp.login(SMTP_USER, SMTP_PASSWORD)
smtp.sendmail(SMTP_USER, to_users, msg.as_string())
smtp.close()