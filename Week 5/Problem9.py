# Initialize an empty list to hold student names
students = []

def display_students():
    """Function to display the current list of students."""
    if students:
        print("Current students in the class:")
        for idx, student in enumerate(students, start=1):
            print(f"{idx}. {student}")
    else:
        print("No students in the class.")

def add_student(name):
    """Function to add a student to the list."""
    students.append(name)
    print(f"{name} has been added to the class.")

def remove_student(name):
    """Function to remove a student from the list."""
    if name in students:
        students.remove(name)
        print(f"{name} has been removed from the class.")
    else:
        print(f"{name} is not in the class.")

def main():
    while True:
        print("1. Add a student")
        print("2. Remove a student")
        print("3. Display students")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            name = input("Enter the student's name to add: ")
            add_student(name)
        elif choice == '2':
            name = input("Enter the student's name to remove: ")
            remove_student(name)
        elif choice == '3':
            display_students()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
main()
