import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.model')

# model = tf.keras.models.load_model('handwritten.model')

####### loss, accuracy parameters of the model##########
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# image_number = 1
# while os.path.isfile(f"digits/dig{image_number}.png"):
#     try:
#         img = cv.imread(f"digits/dig{image_number}.png")[:,:,0]
#         print(f"'i am here {image_number}'")
#         # img = np.invert(np.array([img]))
#
#         img = np.invert(img)
#         prediction = model.predict(img)
#         print(f"I think the num is {np.argmax(prediction)}")
#         # plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.imshow(img[0], cmap=plt.cm.binery)
#         plt.show()
#     except:
#         print("error!")
#     finally:
#         image_number += 1

for image_number in range(1,4):

    image = cv2.imread(f"digits/dig{image_number}.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    input_image = image.reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Make predictions
    predictions = model.predict(input_image)
    predicted_digit = tf.argmax(predictions[0]).numpy()

    print("Predicted digit:", predicted_digit)
    # plt.imshow(image[0], cmap=plt.cm.binery)
    plt.show()
