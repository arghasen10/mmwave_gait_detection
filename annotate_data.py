import csv
from datetime import datetime, timedelta

filename = 'soumik_walking_lr_1.csv'
with open(filename, 'w') as file:
    fieldnames = ['Datetime', 'Activity', 'User', 'Type']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

start_time = datetime.strptime("2024-10-24 17:35:50", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2024-10-24 17:37:22", "%Y-%m-%d %H:%M:%S")

curr_time = start_time
while curr_time < end_time:
    with open(filename, 'a') as file:
        fieldnames = ['Datetime', 'Activity', 'User', 'Type']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'Datetime': curr_time, 'Activity': 'left_leg_up' ,'User': 'soumik', 'Type': 'left-right'})

    curr_time += timedelta(milliseconds=100)
