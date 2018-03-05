import wrds
import  pandas as pd
import datetime
from time import time
import numpy as np
from ConfigParser import SafeConfigParser
import sys
sys.path.append('..')
import os

start_time = time()

# Text file with equity tic names
list_file_path = '../../equity_list/all_us_list.txt'

# Read names
with open(list_file_path) as f:
    content = f.readlines()
    equity_names = [x.strip() for x in content]

equity_names = tuple(["'%s'"%str(i) for i in equity_names])
equity_list = ",".join(equity_names)

# Parse field names for sql query
config_field_names = SafeConfigParser()
config_field_names.read('field_names.ini')
sections = config_field_names.sections()

field_names = {}

for section in sections:
    field_names[section] = config_field_names.get(section,'f')


db = wrds.Connection()

# Get combined data
q_combined = ("select "
                "%s "
                "from compa.funda "
                "where tic in (%s) "
                )%(field_names['combined'],equity_list)

combined_df = db.raw_sql(q_combined)
combined_df.to_csv('combined_all_v2.csv')

# Get others data
q_others = ("select "
                "%s "
                "from compa.funda "
                "where tic in (%s) "
                )%(field_names['others'],equity_list)

others_df = db.raw_sql(q_others)
others_df.to_csv("others_all_v2.csv")

# Get mkt data
q_mkt = ("select "
                "%s "
                "from compa.funda "
                "where tic in (%s) "
                )%(field_names['stock_stats'],equity_list)

mkt_df = db.raw_sql(q_mkt)
mkt_df.to_csv("stock_stats_all_v2.csv")
