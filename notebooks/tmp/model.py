
model = create_model(SHAPE)

# model.compile(loss='mse', optimizer='sgd')
model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=["acc", f1])
