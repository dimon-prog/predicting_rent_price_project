document.addEventListener('DOMContentLoaded', function() {
    const districtSelect = document.getElementById('district');

    for (let i = 1; i <= 23; i++) {
        const option = document.createElement('option');
        const districtNumber = i.toString().padStart(2, '0');
        option.value = i;
        option.textContent = `district ${districtNumber}`;
        districtSelect.appendChild(option);
    }

    districtSelect.value = 10;
});


const form = document.getElementById('prediction-form');
const resultSpan = document.getElementById('prediction-result');

form.addEventListener('submit', function(event) {
    event.preventDefault();


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

            throw new Error(data.error || 'Неизвестная ошибка ответа');
        }
    })
    .catch(error => {
        console.error('Произошла ошибка:', error);
        resultSpan.textContent = 'Ошибка!';
        resultSpan.classList.add('error');
    });
});