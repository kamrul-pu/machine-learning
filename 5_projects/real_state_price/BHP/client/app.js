
function getBathValue() {
    const uiBathrooms = document.getElementsByName("uiBathrooms");
    for (var i in uiBathrooms){
        if (uiBathrooms[i].checked){
            return parseInt(i) + 1
        }
    }
    return -1;
}

function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for(var i in uiBHK) {
        if(uiBHK[i].checked){
            return parseInt(i)+1;
        }
    }
    return -1;
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");
  
    var url = "http://127.0.0.1:5000/predict-home-price"; //Use this if you are NOT using nginx which is first 7 tutorials
    // var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  
    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value
    },function(data, status) {
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
        console.log(status);
    });
  }
function onPageLoad() {
    console.log("document loaded");
    var url = "http://127.0.0.1:5000/get-location-names"; // Use this if you are NOT using nginx which is first 7 tutorials
    // var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
    $.get(url, function (data, status) {
        console.log("got response for get_location_names request");
        if (data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for (var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
}

window.onload = onPageLoad;

// function onPageLoad() {
//     console.log("document loaded");
//     var url = "http://127.0.0.1:5000/get-location-names"; // Endpoint URL for fetching location names
//     // var url = "/api/get_location_names"; // Alternative endpoint URL for use with nginx
    
//     // Make an AJAX GET request using the Fetch API
//     fetch(url)
//         .then(response => {
//             // Check if the response is successful (HTTP status in the range 200-299)
//             if (!response.ok) {
//                 throw new Error('Network response was not ok');
//             }
//             return response.json(); // Parse the JSON response
//         })
//         .then(data => {
//             console.log("got response for get_location_names request", data);
            
//             // Check if valid data is received
//             if (data) {
//                 var locations = data.locations; // Extract the 'locations' array from the response
                
//                 var uiLocations = document.getElementById("uiLocations");
//                 uiLocations.innerHTML = ''; // Clear existing options
                
//                 // Iterate over each location received in the response
//                 locations.forEach(location => {
//                     var opt = new Option(location);
//                     uiLocations.appendChild(opt); // Append the <option> element to the <select> element
//                 });
//             }
//         })
//         .catch(error => {
//             console.error('Error fetching data:', error);
//         });
// }
