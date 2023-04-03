
# import the necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


model = None
history = None
active_model = None
x_train = None
y_train = None
y_unTransformed = None
# load the data
# with open('data.json') as f:
#     data = json.load(f)
# df = pd.DataFrame(data['data'])


# #  split the data into features and target

# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # encode the allergy column

# le = LabelEncoder()
# X[:, 0] = le.fit_transform(X[:, 0])


# # encode the skin concern column

# le = LabelEncoder()
# X[:, 1] = le.fit_transform(X[:, 1])


# # encode the skin sensitivity column

# le = LabelEncoder()

# X[:, 2] = le.fit_transform(X[:, 2])


# # encode the skin tone column

# le = LabelEncoder()
# X[:, 3] = le.fit_transform(X[:, 3])


# # encode the skin type column

# le = LabelEncoder()

# X[:, 4] = le.fit_transform(X[:, 4])


# # encode the target column

# le = LabelEncoder()
# y = le.fit_transform(y)


# # split the data into training and testing sets

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0)


# # scale the data

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


def get_data():
    global x_train
    global y_train
    global y_unTransformed
    with open('data.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])

    #  split the data into features and target

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    y_unTransformed = y

    # encode the allergy column

    le = LabelEncoder()
    X[:, 0] = le.fit_transform(X[:, 0])

    # encode the skin concern column

    le = LabelEncoder()
    X[:, 1] = le.fit_transform(X[:, 1])

    # encode the skin sensitivity column

    le = LabelEncoder()

    X[:, 2] = le.fit_transform(X[:, 2])

    # encode the skin tone column

    le = LabelEncoder()

    X[:, 3] = le.fit_transform(X[:, 3])

    # encode the skin type column

    le = LabelEncoder()

    X[:, 4] = le.fit_transform(X[:, 4])


    # encode the target column

    le = LabelEncoder()

    y = le.fit_transform(y)

    # split the data into training and testing sets

    x_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)
    
    # scale the data

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)

    X_test = sc.transform(X_test)


   

def create_model():
    global model
    global history
    model = model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[5]),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    model.fit(x_train, y_train, epochs=100)

    model.save("./saved_model")

    history = model.history.history


def plot_graphs():
    global history
    plt.plot(history['accuracy'])
    plt.plot(history['loss'])
    plt.title('model accuracy and loss')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()




def predict_product(allergy, skin_concern, skin_sensitivity, skin_tone, skin_type):

    global active_model
    # encode the user input

    # encode the allergy column

    le = LabelEncoder()
    allergy = le.fit_transform([allergy])

    # encode the skin concern column

    le = LabelEncoder()
    skin_concern = le.fit_transform([skin_concern])

    # encode the skin sensitivity column

    le = LabelEncoder()

    skin_sensitivity = le.fit_transform([skin_sensitivity])

    # encode the skin tone column

    le = LabelEncoder()

    skin_tone = le.fit_transform([skin_tone])

    # encode the skin type column

    le = LabelEncoder()

    skin_type = le.fit_transform([skin_type])

    # scale the user input

    sc = StandardScaler()

    allergy = sc.fit_transform(allergy.reshape(-1, 1))

    skin_concern = sc.fit_transform(skin_concern.reshape(-1, 1))

    skin_sensitivity = sc.fit_transform(skin_sensitivity.reshape(-1, 1))

    skin_tone = sc.fit_transform(skin_tone.reshape(-1, 1))

    skin_type = sc.fit_transform(skin_type.reshape(-1, 1))


    user_imput = np.array([allergy[0][0], skin_concern[0][0],
                          skin_sensitivity[0][0], skin_tone[0][0], skin_type[0][0]])

    user_imput = user_imput.reshape(1, -1)

    if active_model is None:
        active_model = keras.models.load_model("./saved_model")
    prediction = active_model.predict(user_imput)

    # decode the prediction

    le = LabelEncoder()

    y = le.fit_transform(y_unTransformed)
    print(le.inverse_transform([np.argmax(prediction)]))

    # print the prediction

    return le.inverse_transform([np.argmax(prediction)])[0]
