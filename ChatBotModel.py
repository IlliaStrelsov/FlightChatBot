from keras import Sequential
from keras.layers import Dense, Dropout


class ChatBotModel:

    @staticmethod
    def build(inputShape, outputShape):
        model = Sequential()
        model.add(Dense(128, input_shape=inputShape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(outputShape, activation="softmax"))

        return model
