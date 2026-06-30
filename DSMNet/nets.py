import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout, concatenate

from autoencoder import Autoencoder


class MTL(tf.keras.Model):
    def __init__(self, net, num_classes, sem_flag=True, norm_flag=True):
        super().__init__()

        self.encoder = net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag
        self.num_classes = int(num_classes)

        self.bn0 = BatchNormalization()

        self.deconv1_sem = Conv2DTranspose(1024, 3, strides=2, padding="same")
        self.deconv2_sem = Conv2DTranspose(512, 3, strides=2, padding="same")
        self.deconv3_sem = Conv2DTranspose(256, 3, strides=2, padding="same")
        self.deconv4_sem = Conv2DTranspose(64, 3, strides=2, padding="same")
        self.deconv5_sem = Conv2DTranspose(32, 3, strides=2, padding="same")
        self.conv_sem = Conv2D(self.num_classes, 3, strides=1, padding="same")

        self.do1_sem = Dropout(0.5)
        self.do2_sem = Dropout(0.5)
        self.do3_sem = Dropout(0.5)
        self.do4_sem = Dropout(0.5)
        self.do5_sem = Dropout(0.5)

        self.deconv1_norm = Conv2DTranspose(1024, 3, strides=2, padding="same")
        self.deconv2_norm = Conv2DTranspose(512, 3, strides=2, padding="same")
        self.deconv3_norm = Conv2DTranspose(256, 3, strides=2, padding="same")
        self.deconv4_norm = Conv2DTranspose(64, 3, strides=2, padding="same")
        self.deconv5_norm = Conv2DTranspose(32, 3, strides=2, padding="same")
        self.conv_norm = Conv2D(3, 3, strides=1, padding="same")

        self.do1_norm = Dropout(0.5)
        self.do2_norm = Dropout(0.5)
        self.do3_norm = Dropout(0.5)
        self.do4_norm = Dropout(0.5)
        self.do5_norm = Dropout(0.5)

        self.deconv1 = Conv2DTranspose(1024, 3, strides=2, padding="same")
        self.deconv2 = Conv2DTranspose(512, 3, strides=2, padding="same")
        self.deconv3 = Conv2DTranspose(256, 3, strides=2, padding="same")
        self.deconv4 = Conv2DTranspose(64, 3, strides=2, padding="same")
        self.deconv5 = Conv2DTranspose(32, 3, strides=2, padding="same")

        self.conv_1 = Conv2D(1024, 3, strides=1, padding="same")
        self.conv_2 = Conv2D(1024, 3, strides=1, padding="same")
        self.conv_3 = Conv2D(512, 3, strides=1, padding="same")
        self.conv_4 = Conv2D(512, 3, strides=1, padding="same")
        self.conv_5 = Conv2D(256, 3, strides=1, padding="same")
        self.conv_6 = Conv2D(256, 3, strides=1, padding="same")
        self.conv_7 = Conv2D(64, 3, strides=1, padding="same")
        self.conv_8 = Conv2D(64, 3, strides=1, padding="same")
        self.conv_9 = Conv2D(32, 3, strides=1, padding="same")
        self.conv_10 = Conv2D(32, 3, strides=1, padding="same")

        self.conv_dsm = Conv2D(1, 3, strides=1, padding="same")

        self.do1 = Dropout(0.5)
        self.do2 = Dropout(0.5)
        self.do3 = Dropout(0.5)
        self.do4 = Dropout(0.5)
        self.do5 = Dropout(0.5)

    def call(self, x, training=True):
        x0 = self.encoder(x)
        x0 = self.bn0(x0, training=training)

        x3_sem = x4_sem = x5_sem = x6_sem = x7_sem = None
        if self.sem_flag:
            x_sem = self.deconv1_sem(x0)
            x3_sem = tf.nn.relu(x_sem)
            x_sem = self.do1_sem(x3_sem, training=training)
            x_sem = self.deconv2_sem(x_sem)
            x4_sem = tf.nn.relu(x_sem)
            x_sem = self.do2_sem(x4_sem, training=training)
            x_sem = self.deconv3_sem(x_sem)
            x5_sem = tf.nn.relu(x_sem)
            x_sem = self.do3_sem(x5_sem, training=training)
            x_sem = self.deconv4_sem(x_sem)
            x6_sem = tf.nn.relu(x_sem)
            x_sem = self.do4_sem(x6_sem, training=training)
            x_sem = self.deconv5_sem(x_sem)
            x7_sem = tf.nn.relu(x_sem)
            x_sem = self.do5_sem(x7_sem, training=training)
            x_sem = self.conv_sem(x_sem)
            x_sem = tf.nn.softmax(x_sem)
        else:
            x_sem = None

        x3_norm = x4_norm = x5_norm = x6_norm = x7_norm = None
        if self.norm_flag:
            x_norm = self.deconv1_norm(x0)
            x3_norm = tf.nn.relu(x_norm)
            x_norm = self.do1_norm(x3_norm, training=training)
            x_norm = self.deconv2_norm(x_norm)
            x4_norm = tf.nn.relu(x_norm)
            x_norm = self.do2_norm(x4_norm, training=training)
            x_norm = self.deconv3_norm(x_norm)
            x5_norm = tf.nn.relu(x_norm)
            x_norm = self.do3_norm(x5_norm, training=training)
            x_norm = self.deconv4_norm(x_norm)
            x6_norm = tf.nn.relu(x_norm)
            x_norm = self.do4_norm(x6_norm, training=training)
            x_norm = self.deconv5_norm(x_norm)
            x7_norm = tf.nn.relu(x_norm)
            x_norm = self.do5_norm(x7_norm, training=training)
            x_norm = self.conv_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
        else:
            x_norm = None

        x = self.deconv1(x0)
        x = tf.nn.relu(x)
        if self.norm_flag:
            x = concatenate([x, x3_norm], axis=3)
        if self.sem_flag:
            x = concatenate([x, x3_sem], axis=3)
        x = self.conv_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.do1(x, training=training)

        x = self.deconv2(x)
        x = tf.nn.relu(x)
        if self.norm_flag:
            x = concatenate([x, x4_norm], axis=3)
        if self.sem_flag:
            x = concatenate([x, x4_sem], axis=3)
        x = self.conv_3(x)
        x = tf.nn.relu(x)
        x = self.conv_4(x)
        x = tf.nn.relu(x)
        x = self.do2(x, training=training)

        x = self.deconv3(x)
        x = tf.nn.relu(x)
        if self.norm_flag:
            x = concatenate([x, x5_norm], axis=3)
        if self.sem_flag:
            x = concatenate([x, x5_sem], axis=3)
        x = self.conv_5(x)
        x = tf.nn.relu(x)
        x = self.conv_6(x)
        x = tf.nn.relu(x)
        x = self.do3(x, training=training)

        x = self.deconv4(x)
        x = tf.nn.relu(x)
        if self.norm_flag:
            x = concatenate([x, x6_norm], axis=3)
        if self.sem_flag:
            x = concatenate([x, x6_sem], axis=3)
        x = self.conv_7(x)
        x = tf.nn.relu(x)
        x = self.conv_8(x)
        x = tf.nn.relu(x)
        x = self.do4(x, training=training)

        x = self.deconv5(x)
        x = tf.nn.relu(x)
        if self.norm_flag:
            x = concatenate([x, x7_norm], axis=3)
        if self.sem_flag:
            x = concatenate([x, x7_sem], axis=3)
        x = self.conv_9(x)
        x = tf.nn.relu(x)
        x = self.conv_10(x)
        x = tf.nn.relu(x)
        x = self.do5(x, training=training)

        x = self.conv_dsm(x)
        return x, x_sem, x_norm
