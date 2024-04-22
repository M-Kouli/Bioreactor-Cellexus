let updateInterval = null; // This will hold the reference to the interval for updating the stopwatch
const stopwatchDisplay = document.getElementById('stopwatch');


// This is to control the sidebar by first defining the DOM elements
sidebar = document.querySelector(".sidebar");
toggle = document.querySelector(".toggle");
// Add an event listener to see if the element with class toggle is clicked, the it toggles the class 'close'
toggle.addEventListener("click", () => {
    sidebar.classList.toggle("close");
});


document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/start', { method: 'POST' });
    const message = { command: 'start' };
    ws.send(JSON.stringify(message));
});

document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop', { method: 'POST' });
    const message = { command: 'stop' };
    ws.send(JSON.stringify(message));
});

// Function to update the stopwatch display based on the current state.
function updateStopwatchDisplay(state) {
    // Check if the stopwatch should be running.
    if (state.running) {
        // If there is an existing update interval, clear it to reset.
        if (updateInterval !== null) {
            clearInterval(updateInterval); // Clear existing interval if there is one
        }

        // Define a function to update the stopwatch display.
        const updateFunction = () => {
            const currentTime = Date.now(); // Get the current time in milliseconds.
            const elapsed = currentTime - state.startTime; // Calculate elapsed time since start.
            displayElapsedTime(elapsed); // Update the display with the elapsed time.
        };

        // Immediately update the stopwatch display once.
        updateFunction();
        // Set up an interval to continuously update the stopwatch display every second.
        updateInterval = setInterval(updateFunction, 1000);
    } else {
        // If the stopwatch is not running, clear any existing update interval.
        if (updateInterval !== null) {
            clearInterval(updateInterval); // Clear the interval.
            updateInterval = null; // Reset the interval variable to indicate no ongoing interval.
        }
        // Reset the stopwatch display to 00:00:00.
        displayElapsedTime(0);
    }
}

// Function to display the elapsed time on the stopwatch.
function displayElapsedTime(elapsedTime) {
    // Calculate seconds, minutes, and hours from elapsed time.
    const seconds = Math.floor((elapsedTime / 1000) % 60); // Convert milliseconds to seconds.
    const minutes = Math.floor((elapsedTime / 60000) % 60); // Convert milliseconds to minutes.
    const hours = Math.floor(elapsedTime / (3600000)); // Convert milliseconds to hours.

    // Format the time units to have two digits and concatenate them as a string.
    const formattedTime = [hours, minutes, seconds].map(unit => `${unit}`.padStart(2, '0')).join(':');

    // Find the stopwatch display element by its ID.
    const stopwatchDisplay = document.getElementById('stopwatch');
    // Set the text content of the stopwatch display to the formatted time.
    stopwatchDisplay.textContent = formattedTime;
}

// Not functioning currently
document.getElementById('logoutBtn').addEventListener('click', function () {
    fetch('/logout', {
        method: 'POST',
    })
        .then(response => {
            // Redirect to login page or show a logged out message
            window.location.href = '/login';
        })
        .catch(error => console.error('Error:', error));
});