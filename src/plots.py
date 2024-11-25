import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose

from src.feature_extraction import extract_features_from_data


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

def average_occupancy_heatmap(train_data):
    # Calculate average occupancy rate by day and hour
    weekly_avg = train_data.mean(axis=1)  # Average across all sensors

    # Reshape to separate days and hours
    weekly_avg = weekly_avg.reshape(-1, 144)  # Reshape to (days, 144)
    num_weeks = weekly_avg.shape[0] // 7
    weekly_avg = weekly_avg[:num_weeks * 7].reshape(num_weeks, 7, 144).mean(axis=0)

    # Further reshape to separate hours
    weekly_avg = weekly_avg.reshape(7, 24, 6).mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(weekly_avg, cmap='YlOrRd', xticklabels=range(24), yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Average Occupancy Rate by Day and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.savefig('src/images/AverageOccupancyHeatmap.png')
    plt.close()

def weekly_occupancy_trends(train_data):
    n_days, n_sensors, n_time_intervals = train_data.shape

    dates = pd.date_range(start="2008-01-01", periods=n_days)

    days_of_week = dates.day_name()

    average_occupancy_per_sensor = train_data.mean(axis=(0, 2))
    top_sensors_indices = np.argsort(-average_occupancy_per_sensor)[:5]
    top_sensors = [f"Sensor {i+1}" for i in top_sensors_indices]

    daily_occupancy = train_data.mean(axis=2)
    daily_occupancy_top_sensors = daily_occupancy[:, top_sensors_indices]

    occupancy_df = pd.DataFrame(daily_occupancy_top_sensors, columns=top_sensors)
    occupancy_df['Day of Week'] = days_of_week.values  # Add day of week

    # Ensure correct order of days
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    occupancy_df['Day of Week'] = pd.Categorical(occupancy_df['Day of Week'], categories=day_order, ordered=True)

    weekly_trends = occupancy_df.groupby('Day of Week', observed=False).mean()

    plt.figure(figsize=(10, 8))
    for sensor in weekly_trends.columns:
        plt.plot(weekly_trends.index, weekly_trends[sensor], label=sensor)

    plt.title("Weekly Occupancy Trends for Top Sensors")
    plt.xlabel("Day of the Week")
    plt.ylabel("Average Occupancy Rate")
    plt.legend(title="Sensors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('src/images/WeeklyOccupancyTrends.png')
    plt.close()

def time_series_decomposition(train_data):
    sensor_data = train_data[:, 0, :].flatten()
    result = seasonal_decompose(sensor_data, model='additive', period=144)
    # Generate timestamps (example for 10-minute intervals over 7 days)
    timestamps = pd.date_range(start="2008-01-01", periods=len(sensor_data), freq="16min")

    # Update the plot to show timestamps
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(timestamps, result.observed)
    plt.title('Observed')
    plt.subplot(412)
    plt.plot(timestamps, result.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(timestamps, result.seasonal)
    plt.title('Seasonal')
    plt.subplot(414)
    plt.plot(timestamps, result.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.savefig('src/images/TimeSeriesDecomposition.png')
    plt.close()
    
def average_daily_patterns_by_zones(train_data, test_data):
    kmeans = KMeans(n_clusters=5, random_state=42)
    X_train, X_test = extract_features_from_data(train_data, test_data)

    clusters = kmeans.fit_predict(X_train)

    cluster_names = {
        0: "Peak Traffic Zones",
        1: "Low-Traffic Areas",
        2: "Transitional Periods",
        3: "Anomalous Zones",
        4: "Dynamic or Mixed Zones"
    }

    plt.figure(figsize=(10, 8))
    for i in range(5):
        cluster_data = train_data[clusters == i].mean(axis=0).mean(axis=0)
        plt.plot(cluster_data, label=f"{cluster_names[i]}")
        
    plt.title('Average Daily Pattern by Zones')
    plt.xlabel('Time (10-minute intervals)')
    plt.ylabel('Average Occupancy Rate')
    plt.legend()
    plt.savefig('src/images/AverageDailyPatternsByZones.png')
    plt.close()