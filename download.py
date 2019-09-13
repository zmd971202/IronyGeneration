import sys
import urllib
import re
import json
import urllib.request

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)


for line in open(sys.argv[1]):
    sid = line.rstrip('\n')
    
    try:
        url = "http://twitter.com/intent/retweet?tweet_id=" + sid
        f = urllib.request.urlopen(url)
        #Thanks to Aniruddha
        html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        jstt = soup.find_all("div","tweet-text")
        tweets = list(set([x.get_text() for x in jstt]))
        print(tweets[0].replace('\n', ' '))
    except Exception as e:
        #print()
        continue
