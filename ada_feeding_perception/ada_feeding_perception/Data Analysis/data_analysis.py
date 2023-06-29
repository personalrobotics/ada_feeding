import csv
import matplotlib.pyplot as plt

"""
This file goes through csv files and creates matplotlib plots and saves them to files
"""


def main(args=None):
    # File names
    files = ['HomeDonut', 'HomeLeaf', 'FaceDonut', 'FaceLeaf', 'PlateDonut', 'PlateLeaf']

    for file in files:
        x_values = []
        y_values = []
        labels = []
        with open(
                r'/home/atharva2/atharvak_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/data/' + file + '.csv',
                'r', newline='') as f:
            f = csv.reader(f)
            i = 0
            for line in f:
                if i != 0:
                    x_values.append(i - 1)
                    y_values.append(int(line[1]))
                    labels.append(line[2])
                i += 1

            y_start = min(y_values)
            y_end = max(y_values)

            for i in range(len(labels)):
                if labels[i] == 'empty':
                    plt.fill_betweenx([y_start, y_end], x_values[i - 1], x_values[i], color="red", alpha=0.2)
                elif labels[i] == 'hand':
                    plt.fill_betweenx([y_start, y_end], x_values[i - 1], x_values[i], color="blue", alpha=0.2)
                else:
                    plt.fill_betweenx([y_start, y_end], x_values[i - 1], x_values[i], color="green", alpha=0.2)

        print(x_values)
        plt.scatter(x_values, y_values, color='black')
        plt.xlabel("Time")
        plt.ylabel("Number of Pixel Values in range")
        plt.title("Number of Pixel Values in range vs. Time (" + file + ")")
        # plt.show()
        plt.savefig(file + "_fig.png")


if __name__ == '__main__':
    main()
