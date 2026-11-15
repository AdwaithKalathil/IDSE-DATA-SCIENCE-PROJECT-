(() => {
  'use strict';

  const form = document.getElementById('car-form');
  const fieldsGrid = document.getElementById('fields-grid');
  const formFields = document.getElementById('form-fields');
  const predictButton = document.getElementById('predict-button');
  const loadError = document.getElementById('load-error');
  const numberFields = new Set(['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']);
  const optionalFields = new Set(['mileage', 'engine', 'max_power', 'seats']);
  const integerFields = new Set(['year', 'km_driven', 'seats']);
  const indianCurrency = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });
  let schema = null;
  let busy = false;

  const exampleCar = {
    year: 2015,
    km_driven: 60000,
    fuel: 'Diesel',
    seller_type: 'Individual',
    transmission: 'Manual',
    owner: 'First Owner',
    mileage: 23.4,
    engine: 1248,
    max_power: 74,
    seats: 5,
    brand: 'Maruti'
  };

  function humanise(name) {
    return name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function appendField(name, spec) {
    const wrapper = document.createElement('div');
    wrapper.className = `field${name === 'brand' ? ' full-width' : ''}`;
    wrapper.dataset.field = name;

    const labelRow = document.createElement('div');
    labelRow.className = 'field-label-row';
    const label = document.createElement('label');
    label.htmlFor = `field-${name}`;
    label.textContent = humanise(name);
    labelRow.append(label);

    if (optionalFields.has(name)) {
      const optional = document.createElement('span');
      optional.className = 'optional-label';
      optional.textContent = 'optional (auto-filled with typical values if blank)';
      labelRow.append(optional);
    }
    wrapper.append(labelRow);

    let control;
    if (Array.isArray(spec.allowed_values)) {
      control = document.createElement('select');
      control.required = true;
      control.add(new Option(`Select ${humanise(name).toLowerCase()}`, ''));
      spec.allowed_values.forEach((value) => control.add(new Option(value, value)));
    } else {
      control = document.createElement('input');
      control.type = 'number';
      control.inputMode = 'decimal';
      control.min = spec.train_min;
      control.max = spec.train_max;
      control.step = integerFields.has(name) ? '1' : 'any';
      control.placeholder = optionalFields.has(name) ? 'Leave blank to auto-fill' : 'Enter value';
      control.required = Boolean(spec.required);
    }
    control.id = `field-${name}`;
    control.name = name;
    control.setAttribute('aria-describedby', `hint-${name} error-${name}`);
    control.addEventListener('input', () => clearFieldError(name));
    control.addEventListener('change', () => clearFieldError(name));
    wrapper.append(control);

    const hint = document.createElement('p');
    hint.className = name === 'year' ? 'year-note' : 'range-hint';
    hint.id = `hint-${name}`;
    if (name === 'year') {
      hint.textContent = `Training data covers ${spec.train_min}–${spec.train_max}; later model years are outside the model's training range.`;
    } else if (spec.train_min !== undefined && spec.train_max !== undefined) {
      const unit = name === 'km_driven' ? ' km' : '';
      hint.textContent = `Training range: ${Number(spec.train_min).toLocaleString('en-IN')}–${Number(spec.train_max).toLocaleString('en-IN')}${unit}`;
    } else if (name === 'mileage') {
      hint.textContent = 'kmpl (or km/kg for CNG/LPG)';
    } else if (name === 'engine') {
      hint.textContent = 'Engine displacement in CC';
    } else if (name === 'max_power') {
      hint.textContent = 'Maximum power in bhp';
    } else if (name === 'seats') {
      hint.textContent = 'Number of seats';
    }
    wrapper.append(hint);

    const error = document.createElement('small');
    error.className = 'field-error';
    error.id = `error-${name}`;
    error.setAttribute('aria-live', 'polite');
    wrapper.append(error);
    fieldsGrid.append(wrapper);
  }

  function clearFieldError(name) {
    const wrapper = fieldsGrid.querySelector(`[data-field="${name}"]`);
    if (!wrapper) return;
    wrapper.classList.remove('invalid');
    wrapper.querySelector('.field-error').textContent = '';
    wrapper.querySelector('input, select').removeAttribute('aria-invalid');
  }

  function setFieldError(name, message) {
    const wrapper = fieldsGrid.querySelector(`[data-field="${name}"]`);
    if (!wrapper) return;
    wrapper.classList.add('invalid');
    wrapper.querySelector('.field-error').textContent = message;
    wrapper.querySelector('input, select').setAttribute('aria-invalid', 'true');
  }

  function clearAllErrors() {
    fieldsGrid.querySelectorAll('.field').forEach((wrapper) => {
      wrapper.classList.remove('invalid');
      wrapper.querySelector('.field-error').textContent = '';
      wrapper.querySelector('input, select').removeAttribute('aria-invalid');
    });
  }

  function readAndValidate() {
    clearAllErrors();
    const car = {};
    let firstInvalid = null;

    Object.entries(schema.input_fields).forEach(([name, spec]) => {
      const control = document.getElementById(`field-${name}`);
      const raw = control.value.trim();
      if (raw === '' && optionalFields.has(name)) {
        car[name] = null;
        return;
      }
      if (raw === '') {
        setFieldError(name, 'This field is required.');
        firstInvalid ||= control;
        return;
      }
      if (numberFields.has(name)) {
        const value = Number(raw);
        if (!Number.isFinite(value)) {
          setFieldError(name, 'Enter a valid number.');
          firstInvalid ||= control;
          return;
        }
        if (value < spec.train_min || value > spec.train_max) {
          const extra = name === 'year' ? ' Model years after 2020 are outside the training data.' : '';
          setFieldError(name, `Enter a value from ${spec.train_min} to ${spec.train_max}.${extra}`);
          firstInvalid ||= control;
          return;
        }
        if (integerFields.has(name) && !Number.isInteger(value)) {
          setFieldError(name, 'Enter a whole number.');
          firstInvalid ||= control;
          return;
        }
        car[name] = value;
        return;
      }
      if (Array.isArray(spec.allowed_values) && !spec.allowed_values.includes(raw)) {
        setFieldError(name, 'Select one of the listed options.');
        firstInvalid ||= control;
        return;
      }
      car[name] = raw;
    });

    if (firstInvalid) {
      firstInvalid.focus();
      return null;
    }
    return car;
  }

  function currency(value) {
    return `₹${indianCurrency.format(value)}`;
  }

  function updateResultStatus(label, success) {
    const indicator = document.querySelector('.result-indicator');
    indicator.replaceChildren();
    const dot = document.createElement('span');
    dot.style.background = success ? '#53b89a' : '#d59b48';
    indicator.append(dot, document.createTextNode(` ${label}`));
  }

  function showResult(result) {
    document.getElementById('empty-result').hidden = true;
    document.getElementById('prediction-result').hidden = false;
    document.getElementById('price-value').textContent = currency(result.price);
    document.getElementById('low-value').textContent = currency(result.low);
    document.getElementById('high-value').textContent = currency(result.high);

    const span = Math.max(1, result.high - result.low);
    const position = Math.max(0, Math.min(100, ((result.price - result.low) / span) * 100));
    document.querySelector('.range-marker').style.left = `${position}%`;

    const warnings = document.getElementById('warnings');
    warnings.replaceChildren();
    if (Array.isArray(result.warnings) && result.warnings.length) {
      warnings.hidden = false;
      result.warnings.forEach((warning) => {
        const item = document.createElement('div');
        item.className = 'warning-item';
        item.textContent = `Note: ${warning}`;
        warnings.append(item);
      });
    } else {
      warnings.hidden = true;
    }
    updateResultStatus('COMPLETE', true);
  }

  function showError(message) {
    document.getElementById('empty-result').hidden = true;
    document.getElementById('prediction-result').hidden = false;
    document.getElementById('price-value').textContent = '—';
    document.getElementById('low-value').textContent = '—';
    document.getElementById('high-value').textContent = '—';
    updateResultStatus('CHECK REQUEST', false);
    const warnings = document.getElementById('warnings');
    warnings.replaceChildren();
    warnings.hidden = false;
    const item = document.createElement('div');
    item.className = 'warning-item';
    item.textContent = message;
    warnings.append(item);
  }

  function setBusy(value) {
    busy = value;
    predictButton.disabled = value || !schema;
    predictButton.classList.toggle('loading', value);
    predictButton.querySelector('.button-label').textContent = value ? 'Calculating estimate' : 'Predict resale value';
    predictButton.querySelector('.button-arrow').textContent = '→';
  }

  async function requestPrediction(car) {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(car)
    });
    let data;
    try {
      data = await response.json();
    } catch (_error) {
      throw new Error('The server returned an unreadable response. Check that the Flask app is running.');
    }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy || !schema) return;
    loadError.hidden = true;
    const car = readAndValidate();
    if (!car) return;
    setBusy(true);
    try {
      const result = await requestPrediction(car);
      showResult(result);
    } catch (error) {
      showError(error.message || 'Prediction failed. Please check the details and try again.');
    } finally {
      setBusy(false);
    }
  });

  document.getElementById('example-button').addEventListener('click', () => {
    if (!schema) return;
    Object.entries(exampleCar).forEach(([name, value]) => {
      const control = document.getElementById(`field-${name}`);
      if (control) control.value = String(value);
      clearFieldError(name);
    });
    loadError.hidden = true;
  });

  document.getElementById('reset-button').addEventListener('click', () => {
    form.reset();
    clearAllErrors();
    loadError.hidden = true;
    document.getElementById('prediction-result').hidden = true;
    document.getElementById('empty-result').hidden = false;
    document.getElementById('warnings').hidden = true;
    const indicator = document.querySelector('.result-indicator');
    indicator.replaceChildren();
    const dot = document.createElement('span');
    dot.style.background = '#a6b1bd';
    indicator.append(dot, document.createTextNode(' READY'));
  });

  async function initialise() {
    try {
      const response = await fetch('/api/schema', { headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error(`Could not load the input schema (${response.status}).`);
      schema = await response.json();
      Object.entries(schema.input_fields).forEach(([name, spec]) => appendField(name, spec));
      formFields.disabled = false;
      predictButton.disabled = false;
    } catch (error) {
      loadError.textContent = `${error.message} Refresh the page after checking the Flask server.`;
      loadError.hidden = false;
    }
  }

  initialise();
})();
