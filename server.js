// This creates the localhost server in which the user and arduino will communicate
// By: Mo Kouli
// Date: 9 Mar 2024

const express = require('express'); // Import the Express framework
const WebSocket = require('ws'); // Import the WebSocket library
const path = require('path'); // Import the Path module for handling and transforming file paths

const session = require('express-session');
const bcrypt = require('bcryptjs');
const PORT = 8080; // Define the port number for the HTTP server to listen on
const app = express(); // Create an instance of Express
let currentTableName = ''; // This will hold the name of the current table for the session
app.use(session({
  secret: 'Your_Secret_Key',
  resave: false,
  saveUninitialized: false,  // change to false to ensure no session is saved without data
  cookie: { secure: true, httpOnly: true }
}));
app.use(express.json());
// Use the index.html page that was created which is stored under public, it is a word that is convertionally used in web development as everything inside of it is inteded to be shown to the public.

const db = require('./database');

app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));
// Not functioning currently
// Create the users table if it doesn't exist
db.run(`CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          password TEXT NOT NULL,
          role TEXT NOT NULL
      )`, (createErr) => {
  if (createErr) {
    console.error('Error creating users table', createErr.message);
  } else {
    // Check if the admin user already exists
    db.get(`SELECT * FROM users WHERE username = ?`, ['admin'], (selectErr, row) => {
      if (selectErr) {
        console.error('Error fetching admin user', selectErr.message);
      } else if (!row) {
        // Admin user does not exist, create it
        const hash = bcrypt.hashSync('rootadmin', 10); // Hash the admin password
        db.run(`INSERT INTO users (username, password, role) VALUES (?, ?, ?)`,
          ['admin', hash, 'admin'], (insertErr) => {
            if (insertErr) {
              console.error('Error inserting admin user', insertErr.message);
            } else {
              console.log('Admin user created successfully.');
            }
          });
      } else {
        console.log('Admin user already exists.');
      }
    });
  }
});
// Not functioning currently
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  // Fetch user from the database
  db.get("SELECT * FROM users WHERE username = ?", [username], (err, user) => {
      if (user && bcrypt.compareSync(password, user.password)) {
          req.session.user = user; // Establish a session
          req.session.loggedin = true;
          console.log("Session started for user:", username);
          req.session.save(err => {
            if (err) {
              console.error("Error saving session:", err);
              return res.status(500).send("Failed to initialize session.");
            }
            res.redirect('/');
          });
      } else {
          return res.status(401).send("Invalid credentials");
      }
  });
});
// Not functioning currently
// Logout route
app.get('/logout', (req, res) => {
  console.log("Destroying session for user:", req.session.user?.username);
  req.session.destroy(err => {
    if (err) {
      return console.error("Session destruction error:", err);
    }
    res.redirect('/login.html');
  });
});
// Not functioning currently
// Middleware to check if the user is authenticated
function isAuthenticated(req, res, next) {
  if (req.session.loggedin) {
      return next();
  } else {
      res.redirect('/login');
  }
}
// Middleware to check for admin role
function isAdmin(req, res, next) {
  if (req.session.user && req.session.user.role === 'admin') return next();
  res.status(403).send('Not authorized.'); // Or redirect as appropriate
}

app.get('/', (req, res) => {
  if (req.session.loggedin) {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
  }
  else{
    res.redirect('/login.html')
  }
});

// Route for the charts page, this is the newly addition to allow for a routing to other html pages
app.get('/charts', isAuthenticated, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'charts.html'));
});
let stopwatchState = { running: false, startTime: null };


//Server setup where you handle routes/endpoints
// For setting a new session up
app.post('/start', (req, res) => {
  timestamp = Date.now();
  currentTableName = `session_${timestamp}`;

  db.run(`CREATE TABLE ${currentTableName} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    humidity REAL,
    temperature REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )`, (err) => {
    if (err) {
      console.error('Error creating table', err.message);
      return res.status(500).send('Error starting new session');
    }
    console.log(`Table ${currentTableName} created for new session`);
    res.send(`New session started, data will be logged to ${currentTableName}`);
  });
});

// For stopping a session
app.post('/stop', (req, res) => {
  if (!currentTableName) {
    return res.status(400).send('No active session to stop.');
  }

  const archivedTableName = `archived_${currentTableName}`;
  const sqlRenameTable = `ALTER TABLE ${currentTableName} RENAME TO ${archivedTableName}`;

  db.run(sqlRenameTable, [], (err) => {
    if (err) {
      console.error('Error archiving table', err.message);
      return res.status(500).send('Error stopping session');
    }
    console.log(`Session table ${currentTableName} archived as ${archivedTableName}`);
    res.send(`Session stopped and table archived as ${archivedTableName}`);

    // Reset currentTableName to indicate no active session
    currentTableName = '';
  });
});

// API endpoint to fetch sensor data
app.get('/api/data', (req, res) => {
  if (currentTableName) {
    // There's an active session, fetch data from the current session's table
    const sql = `SELECT * FROM ${currentTableName} ORDER BY timestamp ASC`;
    db.all(sql, [], (err, rows) => {
      if (err) {
        console.error('Error fetching data from current session table', err.message);
        return res.status(500).json({ error: err.message });
      }
      res.json({
        message: "success",
        data: rows
      });
    });
  } else {
    // No active session. Decide how you want to handle this.
    // Example: Send an empty array or fetch data from the most recent session.
    res.json({
      message: "No active session",
      data: []
    });
  }
});


// Endpoint to list all archived tables
app.get('/api/tables', (req, res) => {
  // Query to select only tables with names starting with 'archived_'
  db.all("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'archived_%'", (err, tables) => {
    if (err) {
      console.error(err);
      res.status(500).send('Error fetching archived tables');
      return;
    }
    res.json(tables);
  });
});

// Endpoint to fetch data from a specific table (for CSV download)
app.get('/api/data/:tableName', (req, res) => {
  const tableName = req.params.tableName;
  db.all(`SELECT * FROM ${tableName}`, (err, rows) => {
    if (err) {
      console.error(err);
      res.status(500).send('Error fetching data');
      return;
    }
    res.json(rows);
  });
});



// Start an HTTP server by listening on the specified PORT, and display a message when the server is running.
// const server = app.listen(PORT, () => console.log(`Server running on http://192.168.43.75:${PORT}`));
const server = app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
// Set up a WebSocket server that attaches itself to the existing HTTP server
const wss = new WebSocket.Server({ server });



// Listen for 'connection' events on the WebSocket server, which shows when the arduino or user are connected to the server
// A function connection is created as a call back, so that whenever an arduino or user connects the websocket detects it and runs the function connection
wss.on('connection', function connection(ws) {
  console.log('Client connected'); // Log a message when a client connects
  // Listen for 'message' events from the user, with the function incoming as a call back function same idea as the call back function above
  ws.on('message', function incoming(message) {
    console.log('received: %s', message); // Log any messages sent from the user, the %s acts as a placeholder for the string that will be displayed
    const event = JSON.parse(message);
    if (event.command === 'start') {
      // Start the stopwatch logic
      stopwatchState = { running: true, startTime: Date.now() };
      broadcastStopwatchState();
    } else if (event.command === 'stop') {
      // Stop the stopwatch logic
      stopwatchState = { running: false, startTime: null };
      broadcastStopwatchState();
    } else if (event.command === 'requestState') {
      broadcastStopwatchState();
    } else if (event.command === 'toggle') {
      wss.clients.forEach(function each(client) {
        // This checks the user is NOT the arduino and if the arduino's connection is still open  
        if (client !== ws && client.readyState === WebSocket.OPEN) {
          client.send('toggle'); // Send the message to the arduino, which will control the state of the LED
        }
      });
    }

    if (!currentTableName) {
      console.log("No active session.");
    }
    else {
      try {
        const data = JSON.parse(message);
        // Use the currentTableName in the INSERT statement
        const insertSql = `INSERT INTO ${currentTableName} (humidity, temperature) VALUES (?, ?)`;
        db.run(insertSql, [data.humidity, data.temperature], (err) => {
          if (err) {
            return console.error('Error inserting data into table', err.message);
          }
          console.log(`Data inserted into ${currentTableName}`);
        });
      } catch (error) {
        console.error('Error parsing message JSON:', error);
      }
    }


    // Allows the user to communicate with the arduino indrectly by routing the message through the server
    // After receiving a message from the user, the server iterates over all connected WebSocket clients (arduinos)
    wss.clients.forEach(function each(client) {
      // This checks the user is NOT the arduino and if the arduino's connection is still open  
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message); // Send the message to the arduino, which will control the state of the LED
      }
    });
  });
});
// To send the current state of the stopwatch to all users
function broadcastStopwatchState() {
  wss.clients.forEach(function each(client) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(stopwatchState));
    }
  });
  console.log("Broadcasting stopwatch state:", JSON.stringify(stopwatchState));
}
