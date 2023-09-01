import csv

if __name__ == "__main__":
    csv_to_read_from = "../Master_with_labels_8-30-23.csv" # make sure to use the right csv file!!!

    with open(csv_to_read_from, 'r') as file:
        reader = csv.reader(file)
        modified_rows = []

        for r in reader:
            if r[3] == "no_food":
                r[4] = 0
            elif r[3] == "food" or r[3] == "hand":
                r[4] = 1
            # elif r[4] == "label":
            #     r.append("binary_label")

            modified_rows.append(r)

    with open(csv_to_read_from, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)
