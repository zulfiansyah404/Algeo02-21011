import tkinter
import numpy as np
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog
import time
from tess import answer
import cv2
from camera import capture_image

start = time.time()

customtkinter.set_appearance_mode("System") # Modes: "System" (standard), "Dark",
"Light"
customtkinter.set_default_color_theme("blue") # Themes: "blue" (standard), "green",
"dark-blue"

global dataset
dataset = ''
global filename
global hasil
global solve
filename = ''
hasil = np.array([[]])
solve = False
global execute_time
execute_time = ''
class App(customtkinter.CTk):
    WIDTH = 1000
    HEIGHT = 520

    def __init__(self):
        
        super().__init__()
        self.title("Tubes Algeo")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing) # call .on_closing() when app gets closed
        self.minsize(App.WIDTH, App.HEIGHT)
        self.maxsize(App.WIDTH, App.HEIGHT)
        self.resizable(False, False)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.frame_left = customtkinter.CTkFrame(master=self,border_width=1,
        width=180,
        corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")
        self.frame_right = customtkinter.CTkFrame(master=self ,width=1000, corner_radius=0)
        self.frame_right.grid(row=0, column = 1, sticky="nswe", padx=10, pady=10)
        # ============ frame_left ============
        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10) # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1) # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20) # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10) # empty row with minsize as spacing
        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
        text="Face Recognition",
        text_font=("Roboto Medium", -16)) # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)
        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
        text="Insert Your Dataset",
        command=self.UploadActionfolder)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)
        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
        text="Insert Your Image", compound="folder",
        command=self.UploadActionImage)
        self.button_2.grid(row=3, column=0, pady=10, padx=10)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
        text="Process", compound="folder",
        command=self.process)
        self.button_3.grid(row=4, column=0, pady=10, padx=10)

        self.button_4 = customtkinter.CTkButton(master=self.frame_left,
        text="Camera", compound="folder",
        command=self.open_camera)
        self.button_4.grid(row=5, column=0, pady=10, padx=10)


        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="AppearanceMode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
        values=["Light", "Dark"],
        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")
  
        # ============ frame_right ============
        # configure grid layout (1x1)
        self.frame_right.grid_rowconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(0, weight=1)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right,
        corner_radius=0, width=1000, fg_color=self.frame_right.fg_color)
        self.frame_info.grid(row=0, column=0, sticky="nswe", pady=10, padx=10)
        
        
        self.label_info_dataset = customtkinter.CTkLabel(master=self.frame_info,
        text="Dataset: ", width=300 , anchor="center", corner_radius=5)
        self.label_info_dataset.grid(row=0, column=0, sticky="w")
        
        self.label_location_dataset = customtkinter.CTkLabel(master=self.frame_info,
        text=dataset, fg_color="#0c5174", width=300 , anchor="center", corner_radius=5)
        self.label_location_dataset.grid(row=1, column=0, sticky="w", pady=10)

        self.frame_info2 = customtkinter.CTkFrame(master=self.frame_right,
        corner_radius=0, width=1000, fg_color=self.frame_right.fg_color)
        self.frame_info2.grid(row=0, column=1, sticky="nswe", pady=10, padx=10)

        self.label_info_image = customtkinter.CTkLabel(master=self.frame_info2,
        text="Image: ", width=320 , anchor="center", corner_radius=5)
        self.label_info_image.grid(row=0, column=0, sticky="w")

        self.label_location_image = customtkinter.CTkLabel(master=self.frame_info2,
        text=filename, fg_color="#0c5174", width=320 , anchor="center", corner_radius=5)
        self.label_location_image.grid(row=1, column=0, sticky="w", pady=10)


        self.frame_picture = customtkinter.CTkFrame(master=self.frame_right, width=320, height=200 ,border_width=10, corner_radius=0, border_color=self.frame_right.bg_color) # font name and size in px
        self.frame_picture.grid(row=1, column=0, sticky="nswe")

        self.frame_dataset = customtkinter.CTkFrame(master=self.frame_right, width=320, height=200, border_width=10, corner_radius=0, border_color=self.frame_right.bg_color) # font name and size in px
        self.frame_dataset.grid(row=1, column=1, sticky="nswe")

        
        self.label_2 = customtkinter.CTkLabel(master=self,
        text="Execution Time : \n" + str(execute_time),
        text_font=("Roboto Medium", -10))
        self.label_2.grid(row=0, column=2, sticky="nswe")


        # Buat label di frame picture
        self.label_image = customtkinter.CTkLabel(master=self.frame_picture,
        text="Image", fg_color="#0c5174", width=300, height=30)
        self.label_image.grid(row=0, column=0, sticky="nswe", pady=10, padx=10)

        self.label_dataset = customtkinter.CTkLabel(master=self.frame_dataset,
        text="Result", fg_color="#0c5174", width=320, height=30)
        self.label_dataset.grid(row=0, column=0, sticky="nswe", pady=10, padx=10)

        # self.frame_image = customtkinter.CTkFrame(master=self.frame_picture, width=320, border_width=10) # font name and size in px 
        print(filename)
        if (filename != ''):
            image = Image.open(filename).resize((256, 256))

            self.photo = ImageTk.PhotoImage(image)

            self.label_image = customtkinter.CTkLabel(master=self.frame_picture,
            image=self.photo, width=256, height=256)

            self.label_image.grid(row=1, column=0, sticky="nswe", pady=10, padx=10)

        if (solve):
            image2 = Image.open("hasil.jpg").resize((256, 256))
            self.photo2 = ImageTk.PhotoImage(image2)

            self.label_image2 = customtkinter.CTkLabel(master=self.frame_dataset,
            image=self.photo2)

            self.label_image2.grid(row=1, column=0, sticky="nswe", pady=50, padx=10)
        # Frame picture akan digunakan untuk menempatkan gambar yang diupload

        # ============ frame_bottom ============
        # configure grid layout (1x1)
        
        
        # Frame label akan digunakan untuk menampilkan hasil dari face recognition

    def process(self):
        global execute_time
        global solve
        self.start_time = time.time()
        print(dataset, filename)
        hasil = answer(dataset, filename)
        execute_time = time.time() - self.start_time
        print(execute_time)
        solve = True
        cv2.imwrite("hasil.jpg", hasil[0])
        image2 = Image.open("hasil.jpg").resize((256, 256))
        self.photo2 = ImageTk.PhotoImage(image2)

        self.label_image2 = customtkinter.CTkLabel(master=self.frame_dataset,
        image=self.photo2)

        self.label_image2.grid(row=1, column=0, sticky="nswe", pady=50, padx=10)

        self.label_2 = customtkinter.CTkLabel(master=self,
        text="Execution Time : \n" + str(execute_time) + "\nResult : \n" + str(self.function_similarity(hasil[2])) + "%",
        text_font=("Roboto Medium", -10))
        self.label_2.grid(row=0, column=2, sticky="nswe")
        
        image = Image.open(filename).resize((256, 256))

        self.photo = ImageTk.PhotoImage(image)

        self.label_image = customtkinter.CTkLabel(master=self.frame_picture,
        image=self.photo, width=256, height=256)

        self.label_image.grid(row=1, column=0, sticky="nswe", pady=10, padx=10)
    def function_similarity(self, similarity):
        return (30000 - similarity) / 30000 * 100
    
    def UploadActionfolder(self):
        global dataset
        temp = filedialog.askdirectory()
        print('Selected: ', temp)
        if (temp != dataset or temp != ""):
            dataset = temp
            self.label_location_dataset.config(text=self.shortNameFile(dataset))

    def update(self):
        self.destroy()
        refresh()

    def UploadActionImage(self):
        global filename
        temp = filedialog.askopenfilename()
        print('Selected: ', temp)
        if (temp != filename or temp != ""):
            filename = temp
            # self.update()
            image2 = Image.open(filename).resize((256, 256))
            self.photo2 = ImageTk.PhotoImage(image2)

            self.label_image = customtkinter.CTkLabel(master=self.frame_picture,
            image=self.photo2)
            
            self.label_image.grid(row=1, column=0, sticky="nswe", pady=50, padx=10)
            self.label_location_image.config(text=self.shortNameFile(filename))

    def shortNameFile(self, name):
        self.N = len(name)
        for i in range(self.N-1, 0, -1):
            if (name[i] == '/' or name[i] == '\\') :
                return name[i+1:self.N]
        return name

    def button_event(self):
        print("Button pressed")
    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
    def on_closing(self, event=0):
        self.destroy()

    def open_camera(self):
        global filename
        capture_image()
        
        image2 = Image.open("potret.jpg").resize((256, 256))
        self.photo2 = ImageTk.PhotoImage(image2)
        
        filename = "potret.jpg"

        self.label_image = customtkinter.CTkLabel(master=self.frame_picture,
        image=self.photo2)
        
        self.label_image.grid(row=1, column=0, sticky="nswe", pady=50, padx=10)
        self.label_location_image.config(text=self.shortNameFile(filename))

        self.label_image.grid(row=1, column=0, sticky="nswe", pady=50, padx=10)
        self.label_location_image.config(text=self.shortNameFile(filename))


def refresh():
    print("sok2")
    app = App()
    print("sok")
    app.mainloop()

if __name__ == "__main__":
    app = App()
    app.mainloop()
    print("selesai")