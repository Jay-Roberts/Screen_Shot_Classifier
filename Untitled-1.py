import argparse
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--a',
type=str)

args = parser.parse_args()
a = args.a
print(type(a))

x = 'st_tmp'
y = x[:-4]
print(y)

"""from google.cloud import storage
client = storage.Client()
for b in client.list_buckets():
    print(b)

bucket = client.get_bucket('gamepics')
stuff10 = list(bucket.list_blobs(prefix='TFRecords/10/train'))
print(stuff10)
print(len(stuff10))
"""
import datetime
st = datetime.datetime.now().time()
print(str(st))
y = list(str(st))
y = [e for e in y if e not in (' ',':','.','-')]
print('y',''.join(y))



