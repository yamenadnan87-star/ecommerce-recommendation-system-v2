function displayProducts(products, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!products || products.length === 0) {
        container.innerHTML = '<p class="no-data">لا توجد منتجات.</p>';
        return;
    }
    let html = '';
    products.forEach((p, idx) => {
        const stars = '⭐'.repeat(Math.round(p.avg_rating)) + '☆'.repeat(5 - Math.round(p.avg_rating));
        html += `
            <div class="product-card">
                <div class="product-badge">${idx+1}</div>
                <h3>منتج ${p.product_id}</h3>
                <div class="product-category"><i class="fas fa-tag"></i> ${p.category}</div>
                <div class="product-price">$${p.price}</div>
                <div class="product-rating">${stars} ${p.avg_rating}/5</div>
            </div>
        `;
    });
    container.innerHTML = html;
}

async function loadRandomProducts() {
    const container = document.getElementById('randomProductsGrid');
    if (!container) return;
    try {
        const res = await fetch('/api/random_products');
        const data = await res.json();
        if (data.products) displayProducts(data.products, 'randomProductsGrid');
    } catch (err) {
        console.error('خطأ في تحميل المنتجات', err);
    }
}

async function loadRecommendations(userId) {
    const gridContainer = document.getElementById('productsGrid');
    if (!gridContainer) return;
    try {
        const [recRes, profileRes] = await Promise.all([
            fetch(`/api/recommendations/${userId}`),
            fetch(`/api/user_profile/${userId}`)
        ]);
        const recData = await recRes.json();
        const profileData = await profileRes.json();
        if (recData.recommendations) {
            displayProducts(recData.recommendations, 'productsGrid');
            const fitnessSpan = document.getElementById('fitnessScore');
            if (fitnessSpan) fitnessSpan.innerText = recData.fitness_score?.toFixed(4) || 'N/A';
        }
        if (profileData) {
            const probSpan = document.getElementById('repurchaseProb');
            if (probSpan) probSpan.innerText = (profileData.repurchase_probability * 100).toFixed(1) + '%';
            const prefSpan = document.getElementById('prefCats');
            if (prefSpan) prefSpan.innerText = profileData.preferred_categories?.join(', ') || 'N/A';
        }
    } catch (err) {
        console.error('خطأ', err);
        gridContainer.innerHTML = '<p class="error">فشل تحميل التوصيات</p>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const loggedIn = document.body.dataset.loggedIn === 'true';
    if (!loggedIn) {
        loadRandomProducts();
    } else {
        const userId = document.body.dataset.userId;
        if (userId) loadRecommendations(parseInt(userId));
    }
    const menuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    if (menuBtn && navLinks) {
        menuBtn.addEventListener('click', () => navLinks.classList.toggle('show'));
    }
});