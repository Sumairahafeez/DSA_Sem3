def RowWiseSum(Mat):
    array = []
    for row in Mat:
        sum = 0
        index = 0
        for e in row:
            sum += e
        array.insert(index,sum)
        index+len(Mat)
    return array    
def ColumnWiseSum(Mat):
    array = []
    for index in range(3):
            sum = 0
            for col in Mat:
                sum += col[index]
            array.insert(index,sum)
    return array   
A = [[1,13,13],[5,11,6],[4,4,9]]
print(ColumnWiseSum(A))
print(RowWiseSum(A))