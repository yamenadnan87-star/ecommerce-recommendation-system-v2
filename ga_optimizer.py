import numpy as np
import random
from data_prep import load_and_prepare_data

class GeneticOptimizer:
    def __init__(self, products_df, user_features, repurchase_predictor, population_size=20, generations=15):
        self.products = products_df
        self.user_features = user_features
        self.predictor = repurchase_predictor
        self.population_size = population_size
        self.generations = generations
        self.all_product_ids = products_df['product_id'].tolist()
        
    def get_user_preferences(self, user_id):
        """استخراج تفضيلات المستخدم"""
        user_row = self.user_features[self.user_features['user_id'] == user_id]
        if len(user_row) == 0:
            return {'fav_category': 'Electronics', 'avg_price': 500, 'avg_rating': 3.0}
        fav_cat = user_row.iloc[0]['fav_category']
        # حساب متوسط السعر المفضل من سلوك المستخدم
        return {'fav_category': fav_cat, 'avg_price': 500, 'avg_rating': user_row.iloc[0]['avg_rating']}
    
    def fitness(self, recommendation, user_id):
        """دالة اللياقة: تحسب جودة التوصية"""
        # الحصول على توقعات إعادة الشراء
        repurchase_prob = self.predictor.predict_repurchase_probability(user_id)
        
        # حساب تشابه الفئات
        prefs = self.get_user_preferences(user_id)
        fav_cat = prefs['fav_category']
        category_match = sum(1 for pid in recommendation if self.products[self.products['product_id'] == pid]['category'].values[0] == fav_cat) / len(recommendation)
        
        # حساب متوسط السعر
        prices = []
        ratings = []
        for pid in recommendation:
            prod = self.products[self.products['product_id'] == pid]
            if len(prod) > 0:
                prices.append(prod.iloc[0]['price'])
                ratings.append(prod.iloc[0].get('avg_rating', 3.0))
        avg_price = np.mean(prices) if prices else 500
        avg_rating = np.mean(ratings) if ratings else 3.0
        
        # تطبيع السعر
        price_score = 1 - min(1, abs(avg_price - prefs['avg_price']) / 1000)
        
        # دالة اللياقة النهائية (مزيج من العوامل)
        fitness_val = (0.4 * repurchase_prob) + (0.3 * category_match) + (0.2 * price_score) + (0.1 * (avg_rating / 5))
        return fitness_val
    
    def create_individual(self, user_id, k=5):
        """إنشاء فرد (قائمة توصيات) عشوائي بناءً على تفضيلات المستخدم"""
        prefs = self.get_user_preferences(user_id)
        # اختيار منتجات من الفئة المفضلة بشكل أكبر
        candidates = self.products[self.products['category'] == prefs['fav_category']]['product_id'].tolist()
        if len(candidates) < k:
            candidates = self.all_product_ids
        return random.sample(candidates, min(k, len(candidates)))
    
    def crossover(self, parent1, parent2):
        """عملية التزاوج (نقطة تقاطع واحدة)"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, individual, mutation_rate=0.2):
        """عملية الطفرة: استبدال منتج بمنتج عشوائي"""
        if random.random() < mutation_rate:
            idx = random.randint(0, len(individual)-1)
            new_product = random.choice(self.all_product_ids)
            # تجنب التكرار
            while new_product in individual:
                new_product = random.choice(self.all_product_ids)
            individual[idx] = new_product
        return individual
    
    def optimize(self, user_id, k=5):
        """تنفيذ الخوارزمية الجينية لتحسين التوصيات لمستخدم معين"""
        # السكان الأولي
        population = [self.create_individual(user_id, k) for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            # حساب اللياقة لكل فرد
            fitness_scores = [self.fitness(ind, user_id) for ind in population]
            
            # اختيار أفضل الأفراد (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            best_individuals = [population[i] for i in sorted_indices[:self.population_size//2]]
            
            # إنشاء الجيل الجديد
            new_population = best_individuals.copy()
            while len(new_population) < self.population_size:
                # اختيار الأبوين
                parent1, parent2 = random.sample(best_individuals, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # أفضل توصية
        final_fitness = [self.fitness(ind, user_id) for ind in population]
        best_idx = np.argmax(final_fitness)
        best_recommendation = population[best_idx]
        return best_recommendation, final_fitness[best_idx]