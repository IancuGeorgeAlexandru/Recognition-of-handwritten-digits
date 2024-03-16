import tkinter as tk
from tkinter import filedialog
from classify import classifyfcn
from PIL import Image, ImageDraw


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        # Dimensiunea întregii pagini
        self.page_width = 800
        self.page_height = 600
        self.root.minsize(400, 300)
        self.root.maxsize(400, 300)

        # Dimensiunea desenului
        self.draw_width = 200
        self.draw_height = 200

        # Creează un frame pentru a încadra canvas-ul și butoanele
        self.frame = tk.Frame(root)
        self.frame.pack()

        # Canvas pentru desen
        self.canvas = tk.Canvas(self.frame, width=self.draw_width, height=self.draw_height, bg="gray")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.drawing = False
        self.old_x = None
        self.old_y = None

        self.line_width = 15  # Grosimea liniei

        # Butoane
        self.save_button = tk.Button(self.frame, text="Salvează", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.frame, text="Șterge", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.number_label = tk.Label(self.frame, text="Număr: 0")
        self.number_label.pack(side=tk.LEFT)

        self.increment_button = tk.Button(self.frame, text="Clasifica", command=self.classify)
        self.increment_button.pack(side=tk.LEFT)

        self.number_value = 0

    def start_drawing(self, event):
        self.drawing = True
        self.old_x = event.x
        self.old_y = event.y

    def paint(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.old_x and self.old_y:
                self.canvas.create_line(self.old_x, self.old_y, x, y, width=self.line_width, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
                self.old_x = x
                self.old_y = y

    def stop_drawing(self, event):
        self.drawing = False
        self.old_x = None
        self.old_y = None

    def save_image(self):
        # Calea completă către locul prestabilit pentru salvare
        file_path = "custom_test_images/test.png"

        # Creează o imagine goală cu dimensiunile canvas-ului
        img = Image.new("RGB", (self.draw_width, self.draw_height), "white")
        draw = ImageDraw.Draw(img)

        # Desenează conținutul canvas-ului pe imagine cu grosimea liniei mai mare
        items = self.canvas.find_all()
        for item in items:
            coords = self.canvas.coords(item)
            color = self.canvas.itemcget(item, "fill")
            draw.line(coords, fill=color, width=self.line_width)

        # Salvează imaginea în format PNG
        img.save(file_path, "png")

    def clear_canvas(self):
        # Șterge tot conținutul de pe canvas
        self.canvas.delete("all")

    def classify(self):
        self.number_value  = classifyfcn("test.png")
        self.number_label.config(text=f"Număr: {self.number_value}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
