
const featuresSection = document.getElementById('features-section');

document.getElementById('animal_form').addEventListener('submit', e => {
    e.preventDefault()
    const formData = new FormData(e.target)
    const data = {}
    formData.forEach((value, key) => {
        data[key] = value
    })
    console.log(data)
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {'Content-Type': 'application/json'}
    })
    .then(resp => resp.json())
    .then(result => {
        console.log(result)
        if (result.prediction === "Find With Features") {
            const message = document.getElementById('msg')
            message.innerHTML = `New animal name. Find class-type based on features.`
            featuresSection.classList.remove('hidden')
        }
        // ✅ Unhide the result section
        const resultContainer = document.getElementById('result')
        resultContainer.classList.remove('hidden')

        const showResult = document.getElementById('result-content')
        showResult.innerHTML = `
            <b>Prediction: </b>${result.prediction}
        `
    })
})


document.getElementById('prediction_form').addEventListener('submit', (e) => {
    e.preventDefault()
    const formData = new FormData(e.target)
    const data = {}
    formData.forEach((value, key) => {
        console.log(key)
        if(key === 'animal_name') {
            data[key] = value.trim()
        } else {
            data[key] = Number(value)
        }
    })
    console.log(data)

    fetch('/predict_features', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {'Content-Type': 'application/json'}
    })
    .then(resp => resp.json())
    .then(result => {
        console.log(result)
        // ✅ Unhide the result section
        const resultContainer = document.getElementById('result')
        resultContainer.classList.remove('hidden')
        
        const showResult = document.getElementById('result-content')
        showResult.innerHTML = `
            <b>Prediction: </b>${result.prediction}
        `
    })

})