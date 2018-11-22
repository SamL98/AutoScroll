import tkinter as tk
#from multiprocessing import Process

#from capture_scroll import perform_capture

class ScrollArea:
    def __init__(self, root):
        frame = tk.Frame(root)
        frame.pack()
        self.padText(frame)
        return

    def padText(self, frame):
        textPad = tk.Frame(frame)
        self.text = tk.Text(textPad, height=50, width=100)

        self.scroll = tk.Scrollbar(textPad)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.text.yview)

        self.text.pack(side=tk.LEFT)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        textPad.pack(side=tk.TOP)
        return

def main():
    root = tk.Tk()
    scrollArea = ScrollArea(root)

    txt = '\n'.join([f'Line {i}' for i in range(100)])
    scrollArea.text.insert(tk.INSERT, txt)

    #proc = Process(target=perform_capture)
    #proc.start()

    root.mainloop()
    #proc.join()

if __name__ == '__main__':
    main()