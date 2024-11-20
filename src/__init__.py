from src.pre_processing import get_train_test_data
from src.model import train_model
from src.plots import daily_average_occupancy, hourly_average_occupancy


def chain_of_execution():
    # Load training and testing data (after preprocessing)
    train_data, test_data, train_labels, test_labels = get_train_test_data()
    train_model(train_data, train_labels, test_data, test_labels)
    

def visualization():
    train_data, test_data, train_labels, test_labels = get_train_test_data()
    daily_average_occupancy(train_data)
    hourly_average_occupancy(train_data)