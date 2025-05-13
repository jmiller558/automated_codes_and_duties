import requests
import base64
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase

def create_message_with_attachment(to, subject, msg_body, file_path):
    """Creates a MIME message with an attachment."""
    message = MIMEMultipart()
    message['to'] = to
    message['subject'] = subject

    msg = MIMEText(msg_body)
    message.attach(msg)

    content_type, encoding = mimetypes.guess_type(file_path)
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'

    main_type, sub_type = content_type.split('/', 1)

    with open(file_path, 'rb') as fp:
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())

    filename = file_path.split('/')[-1]
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

def send_message(access_token, message):
    """Sends the email message using the Gmail API."""
    url = 'https://gmail.googleapis.com/gmail/v1/users/me/messages/send'
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=message)

    if response.status_code == 200:
        print(f"Email sent successfully. Message ID: {response.json().get('id')}")
    else:
        print(f"Error sending email. Status Code: {response.status_code}")
        print(f"Response: {response.text}")