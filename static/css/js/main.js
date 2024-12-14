document.addEventListener('DOMContentLoaded', function() {
    // Add any JavaScript functionality here
    const routeSelect = document.getElementById('route-select');
    if (routeSelect) {
        routeSelect.addEventListener('change', function() {
            loadPredictions(this.value);
        });
    }
});

async function loadPredictions(route) {
    try {
        const response = await fetch(`/api/predictions/${route}`);
        const data = await response.json();
        updatePredictionsTable(data);
    } catch (error) {
        console.error('Error loading predictions:', error);
    }
}