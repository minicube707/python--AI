
import matplotlib.pyplot as plt

def ask_yes_no(question):
    
    while True:
        
        answer = input(question + " (Yes/No)\n").strip().lower()
        
        if answer in {"yes", "y"}:
            return True
        
        if answer in {"no", "n"}:
            return False
        
        print("Please answer Yes or No")
        
def handle_key(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)