import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
# loss, accuracy = model.evaluate(x_test, y_test)

import flwr as fl


class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("fitting")
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        print("evaluating")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_train), {"accuracy": accuracy}


fl.client.start_numpy_client(
    server_address="10.42.0.1:8082",  # 10.42.0.188
    client=MnistClient(),
)
