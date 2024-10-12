students = ['Fatima','Fariah','Alina','david','Daud'] #declare a list
def searchStudent(name):
    try:
        if students.index(name):#if index found print the index
            print(f'{name} found at index {students.index(name)}')
        else:
            print(f'{name} not found')  #print not found
    except Exception as ex:
        print({str(ex)}) 
#main test checking         
searchStudent('Fatima')
searchStudent('Ali')                