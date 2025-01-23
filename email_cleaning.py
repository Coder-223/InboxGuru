from bs4 import BeautifulSoup
import re

def clean_email_content(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    text = soup.get_text()

    text = re.sub(r'\s+', " ", text).strip()

    return text