# attention_lstm_corridor.py
# TensorFlow 2.x / Keras implementation
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt

# -------------------------
# Simple additive attention layer
# -------------------------
class AdditiveAttentionLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)  # for query (decoder state)
        self.W2 = layers.Dense(units)  # for values (encoder outputs)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch, hidden) -> expand to (batch, 1, hidden)
        query_with_timeaxis = tf.expand_dims(query, 1)
        # score: (batch, time, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_timeaxis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, time, 1)
        context_vector = attention_weights * values  # (batch, time, hidden)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden)
        return context_vector, tf.squeeze(attention_weights, -1)  # return weights shape (batch, time)

# -------------------------
# Build model
# -------------------------
def build_att_lstm(input_timesteps, input_features, lstm_units=64, att_units=32, dropout=0.3, l2=1e-4):
    inputs = layers.Input(shape=(input_timesteps, input_features), name='sequence_input')
    # Optionally: a Conv1D or Dense to preprocess features
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inputs)
    x = layers.Dropout(dropout)(x)

    # encoder LSTM returns sequences (for attention)
    encoder_outputs, state_h, state_c = layers.LSTM(
        lstm_units, return_sequences=True, return_state=True,
        kernel_regularizer=regularizers.l2(l2)
    )(x)

    # use last hidden state as query
    query = state_h  # shape (batch, lstm_units)
    # attention
    context_vector, att_weights = AdditiveAttentionLayer(att_units)(query, encoder_outputs)

    # combine context + query
    combined = layers.Concatenate()([context_vector, query])
    dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(combined)
    dense = layers.Dropout(dropout)(dense)

    # Output: you can predict next lat/lon (2 values) OR a corridor suitability score (0..1)
    # Example: predict next movement offset (dx, dy)
    outputs = layers.Dense(2, name='next_offset')(dense)

    model = models.Model(inputs=inputs, outputs=[outputs, att_weights])
    model.compile(optimizer='adam', loss={'next_offset': 'mse'},
                  metrics={'next_offset': 'mae'})
    return model

# -------------------------
# Example data generation (toy) and augmentation
# -------------------------
def augment_track_batch(X, y, jitter_std=0.0001):
    # X: (batch, timesteps, features). Example feature 0,1 are lat/lon.
    X_aug = X.copy()
    noise = np.random.normal(scale=jitter_std, size=X[..., :2].shape)
    X_aug[..., :2] += noise
    return X_aug, y

# -------------------------
# Training loop with callbacks, early stopping, class/sample weights
# -------------------------
if __name__ == '__main__':
    # Toy dataset: replace with real GPS + covariates loading
    N = 1000
    T = 20
    F = 6  # e.g., lat, lon, speed, ndvi, dist_to_road, hour_of_day
    X = np.random.randn(N, T, F).astype(np.float32)
    # target: next movement dx,dy (toy)
    y = np.random.randn(N, 2).astype(np.float32)

    # train/test split
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_att_lstm(input_timesteps=T, input_features=F)
    model.summary()

    # Simple data augmentation via tf.data pipeline
    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(500).batch(batch_size).map(
        lambda xx, yy: tf.numpy_function(
            func=lambda a,b: augment_track_batch(a,b, jitter_std=0.01),
            inp=[xx, yy],
            Tout=[tf.float32, tf.float32]
        ), num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Callbacks: early stopping and model checkpoint
    es = callbacks.EarlyStopping(monitor='val_next_offset_mae', patience=8, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint('best_att_lstm.h5', monitor='val_next_offset_mae', save_best_only=True)

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es, ckpt])

    # Example: get attention weights for a validation sample
    # the model returns two outputs; when predicting we can use a separate model to fetch attention
    att_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)  # last layer is Dense -> careful
    # Simpler: reconstruct small model to output attention layer by name; or extract via predict
    preds = model.predict(X_val[:5])
    next_offset_preds = preds[0]
    attention_weights = preds[1]  # shape (batch, time)

    # Plot attention for the first sample
    sample_att = attention_weights[0]
    plt.plot(sample_att)
    plt.title('Attention weights across time steps (sample 0)')
    plt.xlabel('time step')
    plt.ylabel('attention weight')
    plt.show()