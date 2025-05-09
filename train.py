import argparse
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
args = parser.parse_args()

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

X_train = train.drop(columns=["is_canceled"])
y_train = train["is_canceled"]

X_val = val.drop(columns=["is_canceled"])
y_val = val["is_canceled"]

combined = pd.concat([X_train, X_val])
combined_encoded = pd.get_dummies(combined)

X_train = combined_encoded.iloc[:len(X_train), :]
X_val = combined_encoded.iloc[len(X_train):, :]

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
y_train = y_train.astype("float32")
y_val = y_val.astype("float32")

with open("data/model_columns.txt", "w") as f:
    for col in X_train.columns:
        f.write(col + "\n")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=args.learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

model.save("data/hotel_cancel_model.h5")
