import numpy as np

def get_train_test_data():
    def load_data(file_path):
        with open(file_path, 'r') as file:
            data = []
            for line in file:
                values = line.strip().split(';')
                row = []
                for val in values:
                    row.extend([float(x) for x in val.strip('[]').split()])
                data.append(row)
        return np.array(data)

    def load_labels(file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
            labels = [int(label) for label in content.strip('[]').split()]
        return np.array(labels)

    # Load training and testing data
    train_data = load_data('data/PEMS_train')
    test_data = load_data('data/PEMS_test')

    # Reshape data (963 sensors x 144 intervals)
    train_data = train_data.reshape(-1, 963, 144)
    test_data = test_data.reshape(-1, 963, 144)

    # Load labels
    train_labels = load_labels('data/PEMS_trainlabels')
    test_labels = load_labels('data/PEMS_testlabels')


    # Print shapes to verify
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test labels shape:", test_labels.shape)

    return train_data, test_data, train_labels, test_labels