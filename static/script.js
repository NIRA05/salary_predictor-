const form = document.getElementById('predict-form');
const resultEl = document.getElementById('result');
const metaEl = document.getElementById('model-meta');
const submitBtn = document.getElementById('submit-btn');

function populateSelect(selectId, values) {
  const select = document.getElementById(selectId);
  select.innerHTML = '';
  values.forEach((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

async function loadOptions() {
  const res = await fetch('/api/options');
  const data = await res.json();
  populateSelect('gender', data.inputs.gender);
  populateSelect('education_level', data.inputs.education_level);
  populateSelect('job_title', data.inputs.job_title);
  metaEl.textContent = `${data.model_class} • ${data.feature_count} encoded features`;
}

function showError(message) {
  resultEl.className = 'result-placeholder result-error';
  resultEl.textContent = message;
}

function showResult(value, note) {
  resultEl.className = 'result-placeholder';
  resultEl.innerHTML = `
    <div>
      <div class="result-value">${Number(value).toLocaleString()}</div>
      <div class="result-label">Predicted Value</div>
      <div class="result-label">${note}</div>
    </div>
  `;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Predicting...';

  const payload = {
    age: Number(document.getElementById('age').value),
    years_of_experience: Number(document.getElementById('years_of_experience').value),
    gender: document.getElementById('gender').value,
    education_level: document.getElementById('education_level').value,
    job_title: document.getElementById('job_title').value,
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || 'Prediction request failed.');
    } else {
      showResult(data.predicted_value, data.currency_note);
    }
  } catch (error) {
    showError('Network error. Please try again.');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Predict';
  }
});

loadOptions().catch(() => {
  showError('Failed to load model options.');
});
