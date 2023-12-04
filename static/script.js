function generateResult(response) {
    const toAppend = document.createElement("div");
    const image = document.createElement("img");
    image.src = "png/" + response['image_path'];
    image.width = 500;
    image.height = 500;
    toAppend.appendChild(image);
    
    const caption = document.createElement('p');
    caption.innerHTML = response['caption'];
    toAppend.appendChild(caption);

    const label = document.createElement('p');
    label.innerHTML = "label: " + response['label'];
    toAppend.appendChild(label);
    
    const results = document.getElementById('results');
    results.appendChild(toAppend)
    results.appendChild(document.createElement('br'))

}

function submitSearch() {
    const textQuery = document.getElementById('textInput').value;
    const imageQuery = document.getElementById('imageInput').files[0];
    
    const categories = document.getElementsByClassName('type');
    let checkedBoxes = [];
    for(let i = 0; i < categories.length; i++) {
        if (categories[i].checked) {
            checkedBoxes.push(categories[i].id);
        }
    }

    console.log(checkedBoxes);
    // Create FormData object to handle multipart/form-data
    var formData = new FormData();
    if(textQuery) {
        formData.append('query', textQuery);
        formData.append('type', 'text');
    }
    else if(imageQuery) {
        formData.append('query', imageQuery);
        formData.append('type', 'image');
    }

    if(checkedBoxes) {
        formData.append('types', checkedBoxes)
    }

    fetch('/', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = "";
        data.forEach(element => generateResult(element));
    })
    .catch(error => {
        console.error('Error:', error);
    });
}