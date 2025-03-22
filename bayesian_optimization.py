from skopt import gp_minimize
from tensorflow.keras.optimizers import Adam

def train_with_params(lr, batch_size, l2_val):
    # build and compile model
    model = build_msrcnn()
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train and evaluate (use small epochs for fast tuning)
    history = model.fit(train_generator, epochs=3, validation_data=val_generator)
    return -max(history.history['val_accuracy'])  # minimize negative accuracy

res = gp_minimize(
    func=lambda x: train_with_params(x[0], int(x[1]), x[2]),
    dimensions=[(0.0001, 0.01), (16, 128), (0.0001, 0.01)],
    n_calls=20,
    random_state=42
)
