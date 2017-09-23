def clean_split_dates(row):
    # Initial date contains the current value for the Date column
    initial_date = str(row['Date'])

    # Split initial_date into two elements if "-" is found
    split_date = initial_date.split('-')

    # If a "-"  is found, split_date will contain a list with at least two items
    if len(split_date) > 1:
        final_date = split_date[0]
    # If no "-" is found, split_date will just contain 1 item, the initial_date
    else:
        final_date = initial_date

    return final_date


# Assign the results of "clean_split_dates" to the 'Date' column.
# We want Pandas to go row-wise so we set "axis=1". We would use "axis=0" for column-wise.
artworks_100['Date'] = artworks.apply(lambda row: clean_split_dates(row), axis=1)
artworks_100['Date'].value_counts()