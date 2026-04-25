import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_and_prepare_data():
    """تحميل البيانات وتجهيزها لتدريب النموذج"""
    # تحميل الملفات (تأكد من وجودها في مجلد data/)
    products = pd.read_excel('data/products.xlsx')
    ratings = pd.read_excel('data/ratings.xlsx')
    behavior = pd.read_csv('data/behavior_clean.csv')
    
    # توحيد أسماء الأعمدة (تحويل إلى أحرف صغيرة) لتجنب مشكلة Capitalization
    products.columns = products.columns.str.lower()
    ratings.columns = ratings.columns.str.lower()
    behavior.columns = behavior.columns.str.lower()
    
    # التأكد من وجود الأعمدة المطلوبة
    required_prod_cols = ['product_id', 'category', 'price']
    for col in required_prod_cols:
        if col not in products.columns:
            raise KeyError(f"العمود '{col}' غير موجود في ملف products.xlsx. الأعمدة الموجودة: {list(products.columns)}")
    
    # حساب مقاييس كل مستخدم
    user_features = behavior.groupby('user_id').agg({
        'viewed': 'sum',
        'clicked': 'sum',
        'purchased': 'sum'
    }).reset_index()
    
    # حساب متوسط التقييم لكل مستخدم
    user_avg_rating = ratings.groupby('user_id')['rating'].mean().reset_index()
    user_avg_rating.columns = ['user_id', 'avg_rating']
    user_features = user_features.merge(user_avg_rating, on='user_id', how='left')
    user_features['avg_rating'] = user_features['avg_rating'].fillna(3.0)
    
    # حساب الفئة المفضلة لكل مستخدم من السلوك (المنتجات التي اشتراها)
    # دمج السلوك مع المنتجات للحصول على الفئة
    behavior_with_cat = behavior.merge(products[['product_id', 'category']], on='product_id', how='left')
    
    # تجنب الأخطاء: إذا لم تنجح عملية الدمج بسبب عدم وجود product_id في behavior، نتحقق
    if 'category' not in behavior_with_cat.columns:
        # محاولة بديلة: استخراج الفئة من products مباشرة لكل منتج
        product_cat = products.set_index('product_id')['category'].to_dict()
        behavior_with_cat['category'] = behavior['product_id'].map(product_cat).fillna('Electronics')
    
    # الآن نحسب الفئة الأكثر تكراراً بين المشتريات
    purchased_items = behavior_with_cat[behavior_with_cat['purchased'] == 1]
    if len(purchased_items) > 0:
        fav_cat = purchased_items.groupby('user_id')['category'].agg(
            lambda x: x.mode()[0] if not x.empty else 'Electronics'
        ).reset_index()
    else:
        # إذا لم توجد مشتريات، نستخدم الفئة الأكثر مشاهدة
        viewed_items = behavior_with_cat[behavior_with_cat['viewed'] == 1]
        if len(viewed_items) > 0:
            fav_cat = viewed_items.groupby('user_id')['category'].agg(
                lambda x: x.mode()[0] if not x.empty else 'Electronics'
            ).reset_index()
        else:
            fav_cat = pd.DataFrame({'user_id': user_features['user_id'].unique(), 'category': 'Electronics'})
    
    fav_cat.columns = ['user_id', 'fav_category']
    user_features = user_features.merge(fav_cat, on='user_id', how='left')
    user_features['fav_category'] = user_features['fav_category'].fillna('Electronics')
    
    # ترميز الفئة المفضلة
    le = LabelEncoder()
    user_features['fav_cat_encoded'] = le.fit_transform(user_features['fav_category'])
    
    # تطبيع الميزات العددية
    scaler = MinMaxScaler()
    feature_cols = ['viewed', 'clicked', 'purchased', 'avg_rating', 'fav_cat_encoded']
    user_features[feature_cols] = scaler.fit_transform(user_features[feature_cols])
    
    # إنشاء الهدف: هل سيعيد الشراء؟ إذا اشترى أكثر من منتج واحد
    purchase_count = behavior[behavior['purchased'] == 1].groupby('user_id').size().reset_index(name='purchase_count')
    user_features = user_features.merge(purchase_count, on='user_id', how='left')
    user_features['purchase_count'] = user_features['purchase_count'].fillna(0)
    user_features['repurchase'] = (user_features['purchase_count'] > 1).astype(int)
    
    return user_features, products, scaler, le, behavior, ratings


def get_user_sequence(user_id, behavior, products, max_len=10):
    """إنشاء تسلسل سلوكي لمستخدم معين (لـ LSTM)"""
    user_behavior = behavior[behavior['user_id'] == user_id].sort_values(by='product_id')
    if len(user_behavior) == 0:
        return np.zeros((max_len, 4))
    
    seq = []
    for _, row in user_behavior.iterrows():
        product = products[products['product_id'] == row['product_id']]
        if len(product) > 0:
            price_norm = product.iloc[0]['price'] / 2000  # تطبيع تقريبي
        else:
            price_norm = 0.5
        seq.append([row['viewed'], row['clicked'], row['purchased'], price_norm])
    
    # قص أو حشو التسلسل
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [[0,0,0,0]] * (max_len - len(seq)) + seq
    return np.array(seq)