
import cv2
import sys


from sources.preProcess import PreprocessModel

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


userChoice = int(input("enter 1 to to test, 2 to train.\n"))
print("where the train data will be in directory called /train in same sub-directory and test will be in directory "
      "called /test in same sub-directory")
print("the /test directory will contain the files as follows : \n1.png\n2.png\n3.png\n.....")
print("while the /train directory will contain the files as following")
print("         /train ")
print("            | ")
print("      men <- -> women ")
print("      |          |")
print("      0          0")
print("      1          1")
print("      2          2")
print("      3          3")
print("      4          4")
print("      5          5")
print("and each number from 0 -> 5 is a subdirectory that contains images as follows: \n1.png\n2.png\n3.png\n.....")


#print(str(sys.argv[1]))


# reading the image
img = cv2.imread("25.JPG")

# create object of the class
preprocessModel = PreprocessModel()

# getting the image
Image = preprocessModel.preProcess(img)

# saving the image
cv2.imwrite("test.JPG", Image)


