import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
from data_prep import load_and_prepare_data, get_user_sequence

class RepurchasePredictor:
    def __init__(self, model_path='models/repurchase_model.h5'):
        self.model_path = model_path
        self.model = None
        self.user_features_df = None
        self.products_df = None
        self.scaler = None
        self.le = None
        self.behavior = None
        self.ratings = None
        
    def prepare_sequences(self, user_ids, max_len=10):
        """تحضير التسلسلات لجميع المستخدمين"""
        sequences = []
        targets = []
        for uid in user_ids:
            seq = get_user_sequence(uid, self.behavior, self.products_df, max_len)
            sequences.append(seq)
            target = self.user_features_df[self.user_features_df['user_id'] == uid]['repurchase'].values[0]
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape=(10, 4)):
        """بناء نموذج LSTM مشابه لـ EVMN"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        return model
    
    def train(self, user_features, products, scaler, le, behavior, ratings):
        """تدريب النموذج"""
        self.user_features_df = user_features
        self.products_df = products
        self.scaler = scaler
        self.le = le
        self.behavior = behavior
        self.ratings = ratings
        
        user_ids = user_features['user_id'].tolist()
        X_seq, y = self.prepare_sequences(user_ids)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
        
        # بناء وتدريب النموذج
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=16,
            callbacks=[early_stop],
            verbose=1
        )
        
        # تقييم النموذج
        loss, acc, prec, rec, auc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"نتائج التقييم - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
        
        # حفظ النموذج
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        return history
    
    def predict_repurchase_probability(self, user_id):
        """توقع احتمالية إعادة الشراء لمستخدم معين"""
        if self.model is None:
            self.model = load_model(self.model_path)
        seq = get_user_sequence(user_id, self.behavior, self.products_df)
        seq = np.expand_dims(seq, axis=0)
        prob = self.model.predict(seq, verbose=0)[0][0]
        return float(prob)
    
    def load_pretrained(self):
        """تحميل نموذج مدرب مسبقاً"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            return True
        return False