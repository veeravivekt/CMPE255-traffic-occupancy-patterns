import matplotlib.pyplot as plt
import seaborn as sns

def daily_average_occupancy(train_data):
    daily_avg = train_data.mean(axis=(0, 2))

    plt.figure(figsize=(12, 6))
    sns.heatmap(daily_avg.reshape(1, -1), cmap='YlOrRd', annot=False)
    plt.title('Average Daily Occupancy Rate by Sensor')
    plt.xlabel('Sensor')
    # plt.ylabel('Day')
    plt.gca().yaxis.set_visible(False)  # Hide y-axis ticks completely
    plt.savefig('src/images/DailyAverageOccupancy.png')
    plt.close()

def hourly_average_occupancy(train_data):
    # Calculate hourly average occupancy
    hourly_avg = train_data.mean(axis=(0, 1)).reshape(24, 6).mean(axis=1)

    plt.figure(figsize=(10, 8))
    plt.plot(range(24), hourly_avg)
    plt.title('Average Hourly Occupancy Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Occupancy Rate')
    plt.xticks(range(0, 24, 2))
    plt.grid(True)
    plt.savefig('src/images/HourlyAverageOccupancy.png')
    plt.close()