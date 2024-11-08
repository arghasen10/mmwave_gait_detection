import csv
from datetime import datetime, timedelta

filename = 'soumik_walking_fb_2.csv'

# Open the file once outside the loop
with open(filename, 'w', newline='') as file:
    fieldnames = ['Datetime', 'Activity', 'User', 'Type']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize start and end time
    start_time = datetime.strptime("2024-10-24 17:58:40", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2024-10-24 17:59:45", "%Y-%m-%d %H:%M:%S")

    curr_time = start_time

    # Iterate and write rows to the CSV
    while curr_time < end_time:
        writer.writerow({
            'Datetime': curr_time, 
            'Activity': 'l',
            'User': 'Soumik', 
            'Type': 'left-right'
        })
        curr_time += timedelta(milliseconds=100)
