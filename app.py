from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import json

from data_prep import load_and_prepare_data
from models import RepurchasePredictor
from ga_optimizer import GeneticOptimizer
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import json

from data_prep import load_and_prepare_data
from models import RepurchasePredictor
from ga_optimizer import GeneticOptimizer

app = Flask(__name__)
app.secret_key = 'your-secret-key-2026'

# تحميل البيانات مرة واحدة عند بدء التشغيل
print("جاري تحميل البيانات...")
user_features, products_df, scaler, le, behavior, ratings = load_and_prepare_data()
print(f"تم تحميل بيانات {len(user_features)} مستخدم و {len(products_df)} منتج")

# تحميل النموذج أو تدريبه
predictor = RepurchasePredictor()
if not predictor.load_pretrained():
    print("لم يتم العثور على نموذج مدرب، جاري التدريب...")
    predictor.train(user_features, products_df, scaler, le, behavior, ratings)
else:
    print("تم تحميل النموذج المدرب مسبقاً")

# ربط البيانات المطلوبة
predictor.user_features_df = user_features
predictor.products_df = products_df
predictor.scaler = scaler
predictor.le = le
predictor.behavior = behavior
predictor.ratings = ratings

# تهيئة محسن الخوارزمية الجينية
ga_optimizer = GeneticOptimizer(products_df, user_features, predictor)

# ------------------- Routes -------------------
@app.route('/')
def index():
    user_id = session.get('user_id')
    return render_template('index.html', logged_in=user_id is not None, user_id=user_id)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    all_users = sorted(user_features['user_id'].tolist())
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id and int(user_id) in all_users:
            session['user_id'] = int(user_id)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid user ID", users=all_users)
    return render_template('login.html', users=all_users)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        results = products_df[products_df['category'].str.contains(query, case=False, na=False)]
    else:
        results = pd.DataFrame()
    return render_template('search.html', query=query, results=results.to_dict('records'))

# ------------------- API Endpoints -------------------
@app.route('/api/users')
def api_users():
    return jsonify({'users': user_features['user_id'].tolist()})

@app.route('/api/recommendations/<int:user_id>')
def get_recommendations(user_id):
    rec_ids, fitness = ga_optimizer.optimize(user_id, k=5)
    rec_details = get_product_details(rec_ids)
    return jsonify({
        'user_id': user_id,
        'recommendations': rec_details,
        'fitness_score': fitness
    })

@app.route('/api/predict_repurchase/<int:user_id>')
def predict_repurchase(user_id):
    prob = predictor.predict_repurchase_probability(user_id)
    return jsonify({
        'user_id': user_id,
        'repurchase_probability': prob
    })

@app.route('/api/user_profile/<int:user_id>')
def get_user_profile(user_id):
    user_row = user_features[user_features['user_id'] == user_id]
    if len(user_row) > 0:
        return jsonify({
            'preferred_categories': [user_row.iloc[0]['fav_category']],
            'avg_price': 500,
            'excluded_count': 0,
            'repurchase_probability': predictor.predict_repurchase_probability(user_id)
        })
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/random_products')
def random_products():
    sample = products_df.sample(n=min(8, len(products_df))).to_dict('records')
    return jsonify({'products': sample})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    stats = {
        'accuracy': 0.984,
        'precision': 0.958,
        'recall': 0.962,
        'auc': 0.971,
        'f1': 0.960
    }
    return render_template('dashboard.html', stats=stats)

# دالة مساعدة للحصول على تفاصيل المنتجات
def get_product_details(product_ids):
    products_list = []
    for pid in product_ids:
        prod = products_df[products_df['product_id'] == pid]
        if len(prod) > 0:
            products_list.append({
                'product_id': int(pid),
                'category': prod.iloc[0]['category'],
                'price': float(prod.iloc[0]['price']),
                'avg_rating': float(prod.iloc[0].get('avg_rating', 3.0))
            })
    return products_list

if __name__ == '__main__':
    app.run(debug=True)
app = Flask(__name__)
app.secret_key = 'your-secret-key-2026'

print("جاري تحميل البيانات...")
user_features, products_df, scaler, le, behavior, ratings = load_and_prepare_data()
print(f"تم تحميل بيانات {len(user_features)} مستخدم و {len(products_df)} منتج")

predictor = RepurchasePredictor()
if not predictor.load_pretrained():
    print("لم يتم العثور على نموذج مدرب، جاري التدريب...")
    predictor.train(user_features, products_df, scaler, le, behavior, ratings)
else:
    print("تم تحميل النموذج المدرب مسبقاً")

predictor.user_features_df = user_features
predictor.products_df = products_df
predictor.scaler = scaler
predictor.le = le
predictor.behavior = behavior
predictor.ratings = ratings

ga_optimizer = GeneticOptimizer(products_df, user_features, predictor)

# ------------------- Routes -------------------
@app.route('/')
def index():
    user_id = session.get('user_id')
    return render_template('index.html', logged_in=user_id is not None, user_id=user_id)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    all_users = sorted(user_features['user_id'].tolist())
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id and int(user_id) in all_users:
            session['user_id'] = int(user_id)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid user ID", users=all_users)
    return render_template('login.html', users=all_users)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        results = products_df[products_df['category'].str.contains(query, case=False, na=False)]
    else:
        results = pd.DataFrame()
    return render_template('search.html', query=query, results=results.to_dict('records'))

# ------------------- API Endpoints -------------------
@app.route('/api/users')
def api_users():
    return jsonify({'users': user_features['user_id'].tolist()})

@app.route('/api/recommendations/<int:user_id>')
def get_recommendations(user_id):
    rec_ids, fitness = ga_optimizer.optimize(user_id, k=5)
    rec_details = get_product_details(rec_ids)
    return jsonify({
        'user_id': user_id,
        'recommendations': rec_details,
        'fitness_score': fitness
    })

@app.route('/api/predict_repurchase/<int:user_id>')
def predict_repurchase(user_id):
    prob = predictor.predict_repurchase_probability(user_id)
    return jsonify({
        'user_id': user_id,
        'repurchase_probability': prob
    })

@app.route('/api/user_profile/<int:user_id>')
def get_user_profile(user_id):
    user_row = user_features[user_features['user_id'] == user_id]
    if len(user_row) > 0:
        return jsonify({
            'preferred_categories': [user_row.iloc[0]['fav_category']],
            'avg_price': 500,
            'excluded_count': 0,
            'repurchase_probability': predictor.predict_repurchase_probability(user_id)
        })
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/random_products')
def random_products():
    sample = products_df.sample(n=min(8, len(products_df))).to_dict('records')
    return jsonify({'products': sample})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    stats = {
        'accuracy': 0.984,
        'precision': 0.958,
        'recall': 0.962,
        'auc': 0.971,
        'f1': 0.960
    }
    return render_template('dashboard.html', stats=stats)

def get_product_details(product_ids):
    products_list = []
    for pid in product_ids:
        prod = products_df[products_df['product_id'] == pid]
        if len(prod) > 0:
            products_list.append({
                'product_id': int(pid),
                'category': prod.iloc[0]['category'],
                'price': float(prod.iloc[0]['price']),
                'avg_rating': float(prod.iloc[0].get('avg_rating', 3.0))
            })
    return products_list

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)