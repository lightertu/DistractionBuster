# Author: Shweta

import urllib2
import csv
from bs4 import BeautifulSoup

f = open("C:\Users\shwet\Desktop\CS_wiki.csv", 'rb')
f1 = open("C:\Users\shwet\Desktop\wikiContent.xml", 'a')

reader = csv.reader(f)
id=""
c=0
for row in reader:
	if c==0:
		c+=1
		continue

	id+=row[2]+"|"

	c+=1
	print id

	if(c%50==0):
		url="https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=xml&pageids="+id[:-1]
		req=urllib2.urlopen(url)
		if req.getcode() == 200:
			soup = BeautifulSoup(req.read(), 'html.parser')
			print len(soup.find_all('page'))
			print soup.find_all('page')
			s=soup.find_all('page')
			for si in s:
				f1.write(str(si))
			id=""





