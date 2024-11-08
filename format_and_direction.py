import csv
from datetime import datetime, timedelta

input_filename = 'csv/18_08_46.csv'
output_filename = 'new_csvs/18_08_46.csv'

# Set initial direction based on user input, e.g., 'front' or 'back'
initial_direction = 'left'
current_direction = initial_direction
prev_time = None

# Open the input file with encoding='utf-8-sig' to handle BOM issues
with open(input_filename, 'r', encoding='utf-8-sig') as infile, open(output_filename, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Direction']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Standardize the datetime format with milliseconds
        datetime_str = row['Datetime'].strip()
        if len(datetime_str) == 19:  # Missing milliseconds, e.g., '2024-10-24 18:32:16'
            datetime_str += '.000000'  # Append default milliseconds

        # Parse and reformat datetime to ensure consistent output with milliseconds
        try:
            current_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
            formatted_datetime = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            print(f"Error parsing datetime: {datetime_str}")
            continue

        # Check for time difference and update direction if needed
        if prev_time is not None:
            time_diff = current_time - prev_time
            if time_diff > timedelta(milliseconds=100):
                current_direction = 'left' if current_direction == 'right' else 'right'

        # Update row with the formatted datetime and current direction
        row['Datetime'] = formatted_datetime
        row['Direction'] = current_direction
        writer.writerow(row)

        # Update previous time for the next iteration
        prev_time = current_time

