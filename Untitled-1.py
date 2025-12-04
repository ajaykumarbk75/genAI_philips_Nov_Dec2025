# %%
x = 100
print(x)

# %%
#### --- Introduction to python #####
#Print statements

print("Hey Team - we are learning GenAI & We are Happy!")



# %%
# System info

import sys # its library to get system info
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

# %%
# Variables and Data Types
 
x = 100 # Integer type variable
y = 200.5 # Float type variable. 

z = x + y # Addition operation
print("The value of z is:", z)



# %%
# Varibale
x = 'Student1' # String type variable
print(x) 

# %%
# Type of variable
x = 'Student1' # String type variable
print(type(x))   #type function to get the type of variable

# %%
#casting values 
x = 100 # Integer type variable
y = float(x) #casting integer to float
print(y)
print(type(y))

# %%
# Multiple values assignment in a single line

x, y, z = 10, 20.5, 'GenAI'
print(x)
print(y)
print(z)


# %%
x = y = z = "Good Morning"
print(x)
print(y)
print(z)

# %%
# Multple values 
# Sample list 
students = ["Student1", "Student2", "Student3"]
x, y, z = students
print(x)
print(y)
print(z)


# %%
# String values assignment & contactenation
x = "Students in our colleges"
y = "they are studying GenAI"
z = x + " " + y
print(z)

# %%
# Output variables in single print statement
first_name = "John"
last_name = "Doe"
age = 25
print("First Name:", first_name, ", Last Name:", last_name, ", Age:", age)

# %%
# Output variables using f-string (formatted string literals)
first_name = "John"
last_name = "Doe"
age = 25
print(f"First Name: {first_name},  Last Name: {last_name}, Age: {age}") #f strings statements are faster than normal print statements



# %%
# PYTHON MINI PROJECT 
# Simple Calculator
# Function to add two numbers
def add(x, y):
    return x + y
   
# Function to subtract two numbers
def subtract(x, y):
    return x - y
# Function to multiply two numbers
def multiply(x, y):
    return x * y
# Function to divide two numbers
def divide(x, y):
    return x / y      

print(add(10, 5)) 

# %%
# Creating sample function & calling it 

x = "We are learning GenAI with Philips" # x is global variable 
def display_funct():
    print("Hey Team",","+ x)

display_funct()

# %%
display_funct()


# %%
# Python Numbers 
# Integers, Floats, Complex Numbers

x = 1 # int

y = 2.5 # float

z = 1j # complex number

print(x)
print(y)
print(z)

print(type(x))
print(type(y))
print(type(z))

# %%
# Integer data type 
x = 200
y = -30000000
z = -5654327189999
print(type(x))
print(type(y))
print(type(z))

# %%
# Float data type
x = 20.5
y = -35.67
z = 3.4028235e+38 #
a = 9.10938356e-31
b = 10e4    
c = -5.67e-13
print(type(x))
print(type(y))
print(type(z))
print(type(a))
print(type(b))
print(type(c))


# %%
#List data type - it is mutable (can be changed)
studentslist     = ["Student1", 25, "Student3", "Student4"]
print(studentslist)
print(type(studentslist))

# %%
# Python tuple data type - it is immutable (cannot be changed)
studentstuple = ("Student1", "Student2", "Student3", "Student4")
print(studentstuple)
print(type(studentstuple))

# %%
# Dictionary data type - it is mutable (can be changed)
#Dictionary is key-value pair
#Dictionary use curly braces {}

studentdict = {"Name": "Student1",
               "Age": 22,
               "Course": "GenAI"
              }
print(studentdict)
print(type(studentdict))

# %%
# String with Array loops - 
studentnames = ["Alice", "Bob", "Charlie", "David"]
for name in studentnames:  # name is iterator variable
    print(name)


# %%
# tuple & for loop 
studentstuple = ("Student1", "Student2", "Student3", "Student4")
for studentname in studentstuple:
    print(studentname)


# %%
# Dictionary and print with for loop
studentdict = {
    "Name": "Student1",
    "Age": 22,
    "Course": "GenAI"
}   
for key in studentdict:
    print(key, ":", studentdict[key])   

# --- IGNORE ---

# %%
# string with Arrays - loop
for x in "Hey Team - we are learning GenAI & We are Happy!":
    print(x)

# %%
# string with Arrays - loop
for x in "Hey Team - we are learning GenAI & We are Happy!":
    print(x)

# """ """ quote strings
"""This is a multi-line string.
It can span multiple lines.
You can include line breaks
and indentation as needed."""

print("""This is a multi-line string.
It can span multiple lines.
You can include line breaks
and indentation as needed.""")



# %%
# Bool operations
a = True
b = False
print(type(a))
print(type(b))
print(a and b)  # Logical AND
print(a or b)   # Logical OR
print(not a)    # Logical NOT
print(a != b)   # Logical XOR   

# %%
# Mini Project - simple calculator with keyboard input 
# Get input from user
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
operation = input("Enter operation (+, -, *, /): ")

# Perrform calculation based on operation
if operation == '+':
    result = num1 + num2
elif operation == '-':
    result = num1 - num2
elif operation == '*':
    result = num1 * num2
elif operation == '/':
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Error: Division by zero."
else:
    result = "Error: Invalid operation."
# Display result
print("The result is:", result)


# %%
# Mini Project - Even or Odd number
# Get input from user with validation
try:
    num = int(input("Enter an integer: "))
    # Check if the number is even or odd
    if num % 2 == 0:
        print(f"{num} is an even number.")
    else:
        print(f"{num} is an odd number.")
except ValueError:   #catching error for invalid input
    print("Error: Please enter a valid integer (no decimals or letters).")

# %%
# Mini - Project to do list 
todo = [] # Blank List - where we store our tasks

#Create add task function
def add_task(task):
    todo.append(task)  # append() function to add task to list
    print(f'Task "{task}" added to the to-do list.')

#Create show_task() function
def show_tasks():     
    if not todo:          # checking if list is empty
        print("To-do list is empty.") 
    else:
        print("To-do List:") 
        for idx, task in enumerate(todo, start=1): #enumerate function to get index and task
            print(f"{idx}. {task}")

# Sample usage
add_task("Complete GenAI assignment")
add_task("Attend team meeting")
add_task("Review project proposal")
add_task("Plan weekend trip")
show_tasks()


# %%
# Mini Project - Area of the circle 
from math import pi  # importing pi value from math library
r = float(input("Enter the radius of the circle: "))  # taking radius input from user 

area = pi * r**2  # Area calculation formula 
print(f"The area of the circle with radius {r} is: {area}")
print(pi)

# %%
# Mini _ project - Calender 
import calendar
y = int(input("Enter year: ") )
m = int(input("Enter month: ") )
print(calendar.month(y, m))



# %%
# Import datasets and wotking with dataframes

import pandas as pd    #pandas librray used for  data manipulation and analysis

# Path to the CSV file
file_path = 'data/sample_data.csv'



# %%
pip install pandas

# %%
# Import datasets and wotking with dataframes

import pandas as pd    #pandas librray used for  data manipulation and analysis

# Path to the CSV file
file_path = pd.read_csv('/Users/shashikanthb/Desktop/GenAI_Philips_Nov_2025/Largest_Companies.csv')

#Display the file content 

print(file_path.head()) 



# %%
# dataframe operations 
import pandas as pd    #pandas librray used for  data manipulation and analysis

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data) # DataFrame() function to create dataframe
print(df)

# %%
# Josn file 
import pandas as pd
data2 = pd.read_json('/Users/shashikanthb/Desktop/GenAI_Philips_Nov_2025/brazil_geo.json')
print(data2)
df = pd.DataFrame(data2)
print(df.head())


# %%
# dataframe Merging 

df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
  })
print(df1)

df2 = pd.DataFrame({
    'salary': [70000, 80000, 90000],
    'ID': [1, 2, 3]
})
print(df2)
merged_df = pd.merge(df1, df2, on='ID')  # merging two dataframes on 'ID' column 
print(merged_df)





# %%



