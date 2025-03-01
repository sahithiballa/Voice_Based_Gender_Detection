import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os
#import playsound
import pygame.mixer
from audio_processing import record_to_file, extract_feature, load_model
from array import array

root = tk.Tk()
root.title("Voice Gender Detection")
root.geometry("1620x920")

# Load and set the background image
bg_image = Image.open(r"D:\SAHITHI BALLA\projects\5th_sem\VBGD.png")
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label for the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

title_label = tk.Label(root, text="VOICE BASED GENDER DETECTION BY SAHITHI", bg="white", fg="purple", font=("Comic Sans MS", 24, "bold"))
title_label.place(relx=0.5, rely=0.05, anchor="center")

file = ""  # Placeholder for file path

def select_file():
    global file
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        print("Selected file:", file_path)
        file = file_path
    else:
        print("No file selected.")
        exit()
    
def play_audio():
    #global file
    #playsound.playsound(file)
   

# Initialize the mixer.
   pygame.mixer.init()

# Load the audio file.
   sound = pygame.mixer.Sound(file)

   sound.play()

# Wait for the sound to finish playing.
   while pygame.mixer.get_busy():
       pass

# Quit the mixer.
   pygame.mixer.quit()
   
   sound.set_volume(0.5)

def record_audio():
    global file
    print("Please talk")
    file = "test.wav"
    record_to_file(file)

def show_result_image():
    try:
        result_image = Image.open("result_image.png")
        result_image.show()
    except FileNotFoundError:
        messagebox.showerror("Error", "Result image not found. Analyze audio first.")

def analyze_audio():
    global file
    if not os.path.isfile(file):
        print("Invalid file:", file)
        print("Please talk")
        file = "test.wav"
        record_to_file(file)

    # Extract audio features
    features = extract_feature(file, mel=True).reshape(1, -1)

    # Load the trained model (replace with your model loading logic)
    model = load_model(r"D:\SAHITHI BALLA\projects\5th_sem\sahithi\results\model.h5")

    # Gender prediction (replace with your model prediction logic)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"

    # Display the result using the GUI
    if gender == "male":
        result_image = Image.open(r"D:\SAHITHI BALLA\projects\5th_sem\malee.png")
    else:
        result_image = Image.open(r"D:\SAHITHI BALLA\projects\5th_sem\female.png")

    draw = ImageDraw.Draw(result_image)

    title_text = "Voice Based Gender Detection" 
    gender_text = f"Gender: {gender}\nProbabilities: Male: {male_prob * 100:.2f}% Female: {female_prob * 100:.2f}%"
    font = ImageFont.truetype("arial.ttf", size=20)

    title_text_position = (10, 10)
    gender_text_position = (10, 40)

    draw.text(title_text_position, title_text, fill='black', font=font)
    draw.text(gender_text_position, gender_text, fill='black', font=font)

    result_image.show()
    result_image.save("result_image.png")

# Create and configure GUI components
select_button = tk.Button(root, text="Select File", command=select_file, font=("Dancing Script",13 ,"bold"),bg="PINK",fg="black",cursor="hand2")

# Record button
record_button = tk.Button(root, text="Record Audio", command=record_audio, font=("Dancing Script",13 ,"bold"),bg="PINK",fg="black",cursor="hand2")

# Play button
play_button = tk.Button(root, text="Play Audio", command=play_audio, font=("Dancing Script",13 ,"bold"),bg="PINK",fg="black",cursor="hand2")

# Analyze button
analyze_button = tk.Button(root, text="Analyze Audio", command=analyze_audio, font=("Dancing Script",13 ,"bold"),bg="PINK",fg="black",cursor="hand2")

result_button = tk.Button(root, text="Result", command=show_result_image,font=("Dancing Script",13 ,"bold"),bg="PINK",fg="black",cursor="hand2")

select_button.place(x=300, y=400)
record_button.place(x=550, y=400)
play_button.place(x=800, y=400)
analyze_button.place(x=1050,y=400)
result_button.place(x=750, y=500)

root.mainloop()
