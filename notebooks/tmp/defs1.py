
# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
# https://stackoverflow.com/questions/58931078/how-to-replace-certain-parts-of-a-tensor-on-the-condition-in-keras/58931377#58931377


def f1(y_true, y_pred, dtype="float32"):
    dtype = "float32"
    y_pred = K.cast(y_pred, dtype)
    y_true = K.cast(y_true, dtype)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), dtype)
    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), dtype), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, dtype), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), dtype), axis=0)

    p = K.cast(tp / (tp + fp + K.epsilon()), dtype)
    r = K.cast(tp / (tp + fn + K.epsilon()), dtype)

    diff = 2 * p * r
    suma = p + r + K.epsilon()
    d0 = K.equal(diff, 0)
    s0 = K.equal(suma, 0)
    # sum zeros are replaced by ones on division
    rel_dev = diff / K.switch(s0, K.ones_like(suma), suma)
    rel_dev = K.switch(d0 & s0, K.zeros_like(rel_dev), rel_dev)
    try:
        # ~ is the bitwise complement operator in python which essentially calculates (-x - 1)
        rel_dev = K.switch(-d0 - 1 & s0, K.sign(diff), rel_dev)
    except:
        rel_dev = K.switch(~d0 & s0, K.sign(diff), rel_dev)

    f1 = rel_dev

    return K.cast(K.mean(f1), dtype)


# some basic useless model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # model.add(Dense(28))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(28))
    model.add(Activation("sigmoid"))

    return model
