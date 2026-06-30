import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, concatenate


class Autoencoder(tf.keras.Model):
    """DSMNet refinement network used by test_dsm.py."""

    def __init__(self, random_noise_size=100):
        super().__init__()

        self.conv1_0 = Conv2D(64, 3, padding="same")
        self.conv1 = Conv2D(64, 3, padding="same")
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2_0 = Conv2D(128, 3, padding="same")
        self.conv2 = Conv2D(128, 3, padding="same")
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3_0 = Conv2D(256, 3, padding="same")
        self.conv3 = Conv2D(256, 3, padding="same")
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4_0 = Conv2D(512, 3, padding="same")
        self.conv4 = Conv2D(512, 3, padding="same")
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.conv5_0 = Conv2D(1024, 3, padding="same")
        self.conv5 = Conv2D(1024, 3, padding="same")

        self.up6 = Conv2DTranspose(512, 2, strides=2, padding="same")
        self.conv6_0 = Conv2D(512, 3, padding="same")
        self.conv6 = Conv2D(512, 3, padding="same")

        self.up7 = Conv2DTranspose(256, 2, strides=2, padding="same")
        self.conv7_0 = Conv2D(256, 3, padding="same")
        self.conv7 = Conv2D(256, 3, padding="same")

        self.up8 = Conv2DTranspose(128, 2, strides=2, padding="same")
        self.conv8_0 = Conv2D(128, 3, padding="same")
        self.conv8 = Conv2D(128, 3, padding="same")

        self.up9 = Conv2DTranspose(64, 2, strides=2, padding="same")
        self.conv9_0 = Conv2D(64, 3, padding="same")
        self.conv9 = Conv2D(64, 3, padding="same")

        self.out = Conv2D(1, 3, padding="same")

        self.do1 = Dropout(0.5)
        self.do2 = Dropout(0.5)
        self.do3 = Dropout(0.5)
        self.do4 = Dropout(0.5)

    def call(self, x, training=True):
        x = self.conv1_0(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x_1 = tf.nn.relu(x)
        x = self.pool1(x_1)

        x = self.conv2_0(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x_2 = tf.nn.relu(x)
        x = self.pool2(x_2)

        x = self.conv3_0(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x_3 = tf.nn.relu(x)
        x = self.pool3(x_3)

        x = self.conv4_0(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x_4 = tf.nn.relu(x)
        x = self.pool4(x_4)

        x = self.conv5_0(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = tf.nn.relu(x)

        x = self.up6(x)
        x = tf.nn.relu(x)
        x = concatenate([x_4, x], axis=3)
        x = self.conv6_0(x)
        x = tf.nn.relu(x)
        x = self.conv6(x)
        x = tf.nn.relu(x)
        x = self.do1(x, training=training)

        x = self.up7(x)
        x = tf.nn.relu(x)
        x = concatenate([x_3, x], axis=3)
        x = self.conv7_0(x)
        x = tf.nn.relu(x)
        x = self.conv7(x)
        x = tf.nn.relu(x)
        x = self.do2(x, training=training)

        x = self.up8(x)
        x = tf.nn.relu(x)
        x = concatenate([x_2, x], axis=3)
        x = self.conv8_0(x)
        x = tf.nn.relu(x)
        x = self.conv8(x)
        x = tf.nn.relu(x)
        x = self.do3(x, training=training)

        x = self.up9(x)
        x = tf.nn.relu(x)
        x = concatenate([x_1, x], axis=3)
        x = self.conv9_0(x)
        x = tf.nn.relu(x)
        x = self.conv9(x)
        x = tf.nn.relu(x)
        x = self.do4(x, training=training)

        return self.out(x)
