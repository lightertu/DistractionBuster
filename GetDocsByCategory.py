# Author: Shweta
# Peer review: The rest

import urllib
import csv
from bs4 import BeautifulSoup

inputFile = open("./Simplex/literature.csv", 'r')
outputFile = open("./Simplex/Dump/literature.xml", 'a')

reader = csv.reader(inputFile)

pageIds = ""
counter = 0
for row in reader:
    if counter==0:
        counter += 1
        continue

    pageIds += row[2]+"|"
    counter+=1

    if( counter % 50 == 0 ):
        url="https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=xml&pageids="+pageIds[:-1]
        req = urllib.request.urlopen(url)
        if req.getcode() == 200:
            soup = BeautifulSoup(req.read(), 'html.parser')
            s = soup.find_all('page')

            for si in s:
                outputFile.write(str(si))
            pageIds = ""

if pageIds != "":
    url="https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=xml&pageids="+pageIds[:-1]
    req = urllib.request.urlopen(url)
    if req.getcode() == 200:
        soup = BeautifulSoup(req.read(), 'html.parser')
        s = soup.find_all('page')

        for si in s:
            outputFile.write(str(si))
        pageIds = ""

