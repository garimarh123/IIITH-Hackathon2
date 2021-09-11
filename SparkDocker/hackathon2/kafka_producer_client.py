
from kafka import KafkaProducer
import sys
import re
import pandas as pd
import json
import time
import random

import numpy as np
from numpy.lib import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import json

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pandas import read_csv
import pandas as pd

# import matplotlib.pyplot as plt
import copy

import json
import numpy as np

from common_data_processor import substitute


def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka(topic_name, items):
  count=0
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  for message, key in items:
        print(str(message))
        producer.send(topic_name, json.loads(message)).add_errback(error_callback)                                                                      
        count+=1
  producer.flush()
  print("Wrote {0} messages into topic: {1}".format(count, topic_name))

df = pd.read_csv('data/test.csv')


