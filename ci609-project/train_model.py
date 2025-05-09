import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

from preprocessing import (
    load_and_filter_data,
    handle_missing_values,
    feature_engineering_cumulative,
    scale_features_and_split,
    generate_sequence_data
)

#1.load and preprocess data
df_raw = load_and_filter_data()
df_clean = handle_missing_values(df_raw)
df_feat = feature_engineering_cumulative(df_clean, encode_teams=False)

#encode match result as target variable
label_map = {'H': 0, 'D': 1, 'A': 2}
df_clean['FTR_Code'] = df_clean['FTR'].map(label_map)
df_feat['FTR_Code'] = df_clean['FTR_Code']

#split into tabular features + targets
X_tab_train, X_tab_test, y_train, y_test, scaler = scale_features_and_split(df_feat, target_column='FTR_Code')

#generate sequence inputs for LSTM
seq_data = generate_sequence_data(df_feat, window_size=5)
X_seq_train, X_seq_test = seq_data[X_tab_train.index], seq_data[X_tab_test.index]

#one-hot encode class labels
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

#compute class weights to handle imbalance
classes = np.unique(y_train)
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=classes, y=y_train)))

print("Tabular train/test:", X_tab_train.shape, X_tab_test.shape)
print("Sequence train/test:", X_seq_train.shape, X_seq_test.shape)

#2.define hybrid model architecture
def build_hybrid_model(tabular_dim, sequence_shape):
    #tabular input branch
    tab_input = Input(shape=(tabular_dim,), name="tabular_input")
    x = Dense(64, activation='relu')(tab_input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    #sequence input branch
    seq_input = Input(shape=sequence_shape, name="sequence_input")
    y = LSTM(32)(seq_input)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    #merge and output
    combined = concatenate([x, y])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(3, activation='softmax')(z)

    model = Model(inputs=[tab_input, seq_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#model input dimensions
tab_input_dim = X_tab_train.shape[1]
seq_input_shape = X_seq_train.shape[1:]

#3.K-Fold training with ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tab_train), start=1):
    print(f"\n--- Fold {fold} ---")
    X_tab_tr, X_tab_val = X_tab_train.iloc[train_idx], X_tab_train.iloc[val_idx]
    X_seq_tr, X_seq_val = X_seq_train[train_idx], X_seq_train[val_idx]
    y_tr, y_val = y_train_cat[train_idx], y_train_cat[val_idx]

    model = build_hybrid_model(tab_input_dim, seq_input_shape)
    model.fit(
        [X_tab_tr, X_seq_tr], y_tr,
        validation_data=([X_tab_val, X_seq_val], y_val),
        epochs=200, batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        ],
        class_weight=class_weights,
        verbose=1
    )

    model.save(f"football_hybrid_model_fold_{fold}.h5")
    print(f"Saved fold {fold} model")
    ensemble_models.append(model)

#4.train final model on full training set
final_model = build_hybrid_model(tab_input_dim, seq_input_shape)
final_model.fit(
    [X_tab_train, X_seq_train], y_train_cat,
    validation_split=0.2,
    epochs=200, batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    ],
    class_weight=class_weights,
    verbose=1
)

final_model.save('football_hybrid_model_final.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X_tab_train.columns.tolist(), f)
print("Final model, scaler, and features saved.")

#5.evaluate ensemble on test set ---
print("\nEvaluating ensemble on test set...")
ensemble_preds = [model.predict([X_tab_test, X_seq_test], verbose=0) for model in ensemble_models]
avg_preds = np.mean(ensemble_preds, axis=0)
pred_labels = np.argmax(avg_preds, axis=1)

# metrics
acc = accuracy_score(y_test, pred_labels)
macro_f1 = f1_score(y_test, pred_labels, average='macro')
loss = log_loss(y_test, avg_preds)
cm = confusion_matrix(y_test, pred_labels)

print("\n== Ensemble Evaluation ==")
print(f"Accuracy:     {acc:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Log Loss:     {loss:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, target_names=['Home Win', 'Draw', 'Away Win']))

#debug: show first few raw vectors
print("\nSample prediction vectors:")
for i, vec in enumerate(avg_preds[:5]):
    print(f"Sample {i}: {vec}")
