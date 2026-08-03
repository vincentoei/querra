CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INTEGER,
    department_id INTEGER
);

CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(100)
);

INSERT INTO departments (name, location) VALUES
    ('Engineering', 'HQ'),
    ('Sales', 'Remote');

INSERT INTO employees (name, age, department_id) VALUES
    ('Alice', 30, 1),
    ('Bob', 25, 2);
