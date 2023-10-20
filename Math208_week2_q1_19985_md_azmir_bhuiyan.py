import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_variance(data, sample_mean):
    return sum((x - sample_mean) ** 2 for x in data) / len(data)

def calculate_std_dev(variance):
    return variance ** 0.5

def calculate_z_scores(data, mean, std_dev):
    return [(x - mean) / std_dev for x in data]

def calculate_median(data):
    data_sorted = sorted(data)
    n = len(data_sorted)
    midpoint = n // 2
    if n % 2 == 1:
        return data_sorted[midpoint]
    else:
        return (data_sorted[midpoint - 1] + data_sorted[midpoint]) / 2

def calculate_quartiles(data):
    data_sorted = sorted(data)
    n = len(data_sorted)
    midpoint = n // 2

    if n % 2 == 1:
        lower_half = data_sorted[:midpoint]
        upper_half = data_sorted[midpoint+1:]
    else:
        lower_half = data_sorted[:midpoint]
        upper_half = data_sorted[midpoint:]

    q1 = calculate_median(lower_half)
    q3 = calculate_median(upper_half)

    return q1, q3

def plot_boxplots(glucose, blood_pressure):
    fig, ax = plt.subplots()
    ax.boxplot([glucose, blood_pressure], labels=['Glucose', 'Blood Pressure'])
    ax.set_title('Glucose vs Blood Pressure')
    plt.show()

def main():
    # Specify the path to your file
    file_path = 'original_diabetes.xlsx'

    # Read data
    df = read_data(file_path)

    # Extract 'Glucose' and 'BloodPressure' data
    glucose = df['Glucose'].dropna().tolist()
    blood_pressure = df['BloodPressure'].dropna().tolist()

    # Calculate mean, variance, and standard deviation
    glucose_mean = calculate_mean(glucose)
    glucose_variance = calculate_variance(glucose, glucose_mean)
    glucose_std_dev = calculate_std_dev(glucose_variance)
    blood_pressure_mean = calculate_mean(blood_pressure)
    blood_pressure_variance = calculate_variance(blood_pressure, blood_pressure_mean)
    blood_pressure_std_dev = calculate_std_dev(blood_pressure_variance)

    # Calculate z-scores
    glucose_z_scores = calculate_z_scores(glucose, glucose_mean, glucose_std_dev)
    blood_pressure_z_scores = calculate_z_scores(blood_pressure, blood_pressure_mean, blood_pressure_std_dev)

    # Calculate median
    glucose_median = calculate_median(glucose)
    blood_pressure_median = calculate_median(blood_pressure)

    # Calculate Q1 and Q3
    glucose_q1, glucose_q3 = calculate_quartiles(glucose)
    blood_pressure_q1, blood_pressure_q3 = calculate_quartiles(blood_pressure)

    # Output the calculations
    print("Glucose: mean = {}, variance = {}, std_dev = {}, median = {}, Q1 = {}, Q3 = {}".format(
        glucose_mean, glucose_variance, glucose_std_dev, glucose_median, glucose_q1, glucose_q3))
    print("Blood Pressure: mean = {}, variance = {}, std_dev = {}, median = {}, Q1 = {}, Q3 = {}".format(
        blood_pressure_mean, blood_pressure_variance, blood_pressure_std_dev, blood_pressure_median, blood_pressure_q1, blood_pressure_q3))

    # Plot boxplots
    plot_boxplots(glucose, blood_pressure)

if __name__ == "__main__":
    main()
