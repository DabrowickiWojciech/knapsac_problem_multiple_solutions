import sqlite3
import random

# Variables
num_items = 40  # Adjust as needed (value of items generated in db)

# Connect to the database (creates it if it doesn't exist)
conn = sqlite3.connect('items_list.db')
c = conn.cursor()

# Create items table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS items
             (id INTEGER PRIMARY KEY, weight INTEGER, value INTEGER, space INTEGER)''')

# Generate a large number of items
items = [(i+1, random.randint(1, 5), random.randint(1, 500), random.randint(1,8)) for i in range(num_items)]

# Insert items into the database
c.executemany("INSERT INTO items (id, weight, value, space) VALUES (?, ?, ?, ?)", items)

# Commit changes and close connection
conn.commit()
conn.close()