import tkinter as tk
from PIL import ImageTk, Image
import cv2
import threading


def camThread():
    color = []
    cap = cv2.VideoCapture(1)
    panel = None

    while True:
        ret, color = cap.read()
        if(color != []):
            cv2.imshow("color", color)
            image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            if panel is None:
                panel = tk.Label(image=image)
                panel.image = image
                panel.pack(side="left")
            else:
                panel.configure(image=image)
                panel.image = image
            cv2.waitKey(1)

if __name__=='__main__' :
    thread_img = threading.Thread(target=camThread, args=())
    thread_img.start()

    root = tk.Tk()
    root.mainloop()
