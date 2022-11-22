# Import the library
from tkinter import *
from tkinter import filedialog

# Create an instance of window
win=Tk()

# Set the geometry of the window
win.geometry("700x300")

# Create a label
Label(win, text="Click the button to open a dialog", font='Arial 16 bold').pack(pady=15)

# Function to open a file in the system
def open_file():
   filepath = filedialog.askopenfilename(title="Open a Text File", filetypes=(("text    files","*.txt"), ("all files","*.*")))
   file = open(filepath,'r')
   print(file.read())
   file.close()

# Create a button to trigger the dialog
button = Button(win, text="Open", command=open_file)
button.pack()

win.mainloop()