import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import classification_report
plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

frame_stack = 5

def StackFrames(data, labels, frame_stack=4):
    max_index = data.shape[0] - frame_stack
    stacked_data = np.array([data[i:i + frame_stack] for i in range(max_index)])
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_data, new_labels

def split_dataset(data, label):
    np.random.seed(101)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def get_dataset(final_df,frame_stack):
    final_df = final_df[['datetime','range','rangedoppler','User']]
    data = final_df[['range','rangedoppler']].values
    labels = final_df['User'].values
    # data, label = StackFrames(data, labels, frame_stack)
    labelEconder = LabelEncoder()
    label = labelEconder.fit_transform(labels)
    X_train, X_test, y_train, y_test = split_dataset(data, label)
    X_train, y_train = StackFrames(X_train, y_train, frame_stack)
    X_test, y_test = StackFrames(X_test, y_test, frame_stack)
    rp_train = np.expand_dims(np.array(X_train[:,:,0].tolist()).transpose((0,2,1)),axis=2)
    dop_train = np.array(X_train[:,:,1].tolist()).transpose((0,2,3,1))
    print(dop_train.shape)
    rp_test = np.expand_dims(np.array(X_test[:,:,0].tolist()).transpose((0,2,1)), axis=2)
    dop_test = np.array(X_test[:,:,1].tolist()).transpose((0,2,3,1))
    dop_train = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_train = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
    return rp_train, dop_train, rp_test, dop_test, y_train, y_test


def get_cnn2d():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(182, 256, 5)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn2d')
    return model2d

def get_cnn1d():
    model1d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 1), (2, 1), padding="valid", activation='relu', input_shape=(256, 1, 5)),
        tf.keras.layers.Conv2D(64, (8, 1), (2, 1), padding="valid", activation='relu'),
        tf.keras.layers.Conv2D(96, (4, 1), (2, 1), padding="valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn1d')
    return model1d

def get_dda_classifier():
    ann = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l2', input_shape=(224,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ], name='dda_classifier')
    return ann

def featureExtractor(X_1, X_2, cnn2d, cnn1d):
    emb1 = cnn2d(X_1)
    emb2 = cnn1d(X_2)
    return tf.keras.layers.Concatenate(axis=1)([emb1, emb2])



def connect_feature_embeddings():
    cnn2d = get_cnn2d()
    cnn1d = get_cnn1d()
    dda = get_dda_classifier()
    x1, x2= get_feature_shape()
    # Connect
    emb = featureExtractor(x1, x2, cnn2d, cnn1d)
    out_da = dda(emb)
    return x1, x2, out_da

def get_feature_shape():
    X_1 = tf.keras.layers.Input(shape=(182, 256, 5))
    X_2 = tf.keras.layers.Input(shape=(256, 1, 5))
    return X_1, X_2


def get_fused_cnn_model():
    X_1, X_2, out_da = connect_feature_embeddings()
    model = tf.keras.Model(inputs=[X_1, X_2], outputs=[out_da], name='Fused_da_model')
    print(model.summary())
    return model

def train_cnn(model, dop_train_s, rp_train_s, y_train, epochs=100):
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    print('model compile')
    history = \
        model.fit(
            [dop_train_s, rp_train_s],
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
        )
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend(ncols=2)
    plt.show()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(ncols=2)
    plt.show()
    return model


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_model(model, X_test_dop, X_test_rp, y_test):
    y_pred = model.predict([X_test_dop, X_test_rp])
    y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    print("Classification Report:\n", classification_report(y_test, y_pred))

    labels = np.unique(y_test)
    plot_confusion_matrix(y_test, y_pred, labels)

model = get_fused_cnn_model()

# final_df = pd.read_pickle('merged_df.pkl')
# rp_train, dop_train, rp_test, dop_test, y_train, y_test = get_dataset(final_df, frame_stack)

# with open('train_test_data.pkl', 'wb') as f:
#     pickle.dump((rp_train, dop_train, rp_test, dop_test, y_train, y_test), f)


with open('train_test_data.pkl', 'rb') as f:
    rp_train, dop_train, rp_test, dop_test, y_train, y_test = pickle.load(f)

model = train_cnn(model, dop_train, rp_train, y_train)
evaluate_model(model, dop_test, rp_test, y_test)

