#!/bin/sh

zip $1 hlt/*.py MyBot.py  # zip $TARGET.zip $SOURCE1 [$SOURCE2, ...]
./hlt_client/client.py bot -b $1.zip
mv $1.zip ARK/
cd ARK/
unzip $1.zip -d $1
