import tkinter
from tkinter import filedialog
import tkinter.messagebox
import customtkinter
customtkinter.set_appearance_mode("System") # Modes: "System" (standard), "Dark",
"Light"
customtkinter.set_default_color_theme("blue") # Themes: "blue" (standard), "green",
"dark-blue"

global dataset
dataset = ''
global filename
filename = 'Gambar'

class App(customtkinter.CTk):
    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        
        super().__init__()
        print(filename)
        self.title("Tubes Algeo")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing) # call .on_closing() when app gets closed
        # ============ create two frames ============
        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.frame_left = customtkinter.CTkFrame(master=self,
        width=180,
        corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)
        # ============ frame_left ============
        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10) # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1) # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20) # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10) # empty row with minsize as spacing
        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
        text="Tubes Algeo 2",
        text_font=("Roboto Medium", -16)) # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)
        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
        text="Upload",
        command=self.UploadAction)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)
        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
        text="Aksi",
        command=self.button_event)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)
        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="AppearanceMode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
        values=["Light", "Dark"],
        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")
        # ============ frame_right ============
        # configure grid layout (3x7)
        self.frame_right.rowconfigure((1, 1, 1, 1), weight=1)
        self.frame_right.rowconfigure(0, weight=1)
        self.frame_right.columnconfigure((1, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)
        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=2, pady=20, padx=20,
        sticky="nsew")
        # ============ frame_info ============
        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
        text="Gambar" ,
        corner_radius=6, # <- custom corner radius
        fg_color=("white", "gray38"), # <- custom tuple-color
        justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe")
        
        self.frame_info2 = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info2.grid(row=1, column=0, columnspan=2, rowspan=2, pady=20, padx=20,
        sticky="nsew")
        # ============ frame_info ============
        # configure grid layout (1x1)
        self.frame_info2.rowconfigure(0, weight=1)
        self.frame_info2.columnconfigure(0, weight=1)

        self.label_info_2 = customtkinter.CTkLabel(master=self.frame_info2,
        text="Gambar 2" ,
        fg_color=("white", "gray38"), # <- custom tuple-color
        justify=tkinter.RIGHT)
        self.label_info_2.grid(column=0, row=0, sticky="nwe")

        # ============ frame_right ============
  
        
        # self.check_box_1.configure(state=tkinter.DISABLED, text="CheckBox disabled")
        # self.check_box_2.select()
    def UploadAction(self):
        global filename
        temp = filedialog.askopenfilename()
        print('Selected: ', temp)
        if (temp != filename and temp != ""):
            filename = temp
            self.destroy()
            refresh()


    def     button_event(self):
        print("Button pressed")
    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
    def on_closing(self, event=0):
        self.destroy()

def refresh():
    app = App()
    print("sok")
    app.mainloop()

if __name__ == "__main__":
    app = App()
    app.mainloop()
    print("selesai")