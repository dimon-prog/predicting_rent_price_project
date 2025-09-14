// script.js

// --- Первоначальная настройка страницы ---
document.addEventListener('DOMContentLoaded', function() {
    const districtSelect = document.getElementById('district');
    // Динамически создаем опции для выпадающего списка районов
    for (let i = 2; i <= 23; i++) {
        const option = document.createElement('option');
        const districtNumber = i.toString().padStart(2, '0'); // 2 -> "02"
        option.value = i;
        option.textContent = `district ${districtNumber}`;
        districtSelect.appendChild(option);
    }
    // Устанавливаем значение по умолчанию
    districtSelect.value = 10;
});

// --- Обработка отправки формы ---
const form = document.getElementById('prediction-form');
const resultSpan = document.getElementById('prediction-result');

form.addEventListener('submit', function(event) {
    event.preventDefault();

    // Собираем данные из формы в простом, человеческом формате
    const formData = {
        area_sqm: parseFloat(document.getElementById('area_sqm').value),
        rooms: parseFloat(document.getElementById('rooms').value),
        district: parseInt(document.getElementById('district').value),
        has_balcony: document.getElementById('has_balcony').checked, // .checked возвращает true/false
        has_terrace: document.getElementById('has_terrace').checked,
        is_furnished: document.getElementById('is_furnished').checked,
        is_social_housing: document.getElementById('is_social_housing').checked
    };

    const apiUrl = 'http://127.0.0.1:8000/predict';

    // Отправляем данные на сервер
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) { throw new Error(`Ошибка сети: ${response.status}`); }
        return response.json();
    })
    .then(data => {
        if (data.predicted_rent_price) {
            resultSpan.textContent = data.predicted_rent_price;
            resultSpan.classList.remove('error');
        } else {
            // Если сервер вернул ошибку, например, "модель не найдена"
            throw new Error(data.error || 'Неизвестная ошибка ответа');
        }
    })
    .catch(error => {
        console.error('Произошла ошибка:', error);
        resultSpan.textContent = 'Ошибка!';
        resultSpan.classList.add('error');
    });
});