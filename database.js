const sqlite3 = require('sqlite3').verbose();
const dbName = 'bioreactor_data.db'; // Name of your persistent database file
const db = new sqlite3.Database(dbName, sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE, (err) => {
  if (err) {
    console.error('Error opening database', err.message);
  } else {
    console.log('Connected to the SQLite database.');
  }
});

module.exports = db;
