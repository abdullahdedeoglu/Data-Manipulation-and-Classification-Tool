#Abdullah Dedeoglu 2001001046

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas 
from pandastable import Table
import numpy
from DataClassification import *

def import_data(data, tv1):
    tv1["column"]=list(data.columns)
    tv1["show"] = "headings"

    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    data_rows = data.to_numpy().tolist()

    for row in data_rows:
        tv1.insert("", "end", values=row)
    return None

def message_box_for_fill(data):
    def fill_all_and_close():
        fill_all(data)
        custom_dialog.destroy()

    def fill_column_and_close():
        fill_column(data)
        custom_dialog.destroy()

    custom_dialog = tk.Toplevel()
    custom_dialog.title("Custom Dialog")
    custom_dialog.geometry("250x150")

    label = tk.Label(custom_dialog, text="What is the action you want to take?")
    label.pack()

    all_button = tk.Button(custom_dialog, text="All Data", command=lambda: fill_all_and_close())
    all_button.place(x=10, y=30)

    column_button = tk.Button(custom_dialog, text="Only One Column", command=lambda: fill_column_and_close())
    column_button.place(x=10, y=60)
    
def fill_all(data):
    filled_data=fill_missing_all(data)
    filled_data=convert_to_numeric(filled_data)

    tv1.delete(*tv1.get_children())
    tv1["column"]=list(filled_data.columns)
    tv1["show"] = "headings"

    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    data_rows = filled_data.to_numpy().tolist()

    for row in data_rows:
        tv1.insert("", "end", values=row)
    return filled_data

def fill_column(data):
    label = tk.Label(left_frame, text="Column Name=> ")
    label.place(x=20, y=260)

    column_name = tk.Entry(left_frame)
    column_name.place(x=20, y=280)

    def fill_missing():

        selected_column = column_name.get()

        if selected_column not in data.columns:
            messagebox.showwarning("Warning", f"Column '{selected_column}' not found in the dataset!")
            return
        filled_data = fill_missing_column(data, selected_column)
        filled_data = convert_to_numeric(filled_data)
        selected_data=filled_data[selected_column]
        tv2.delete(*tv2.get_children())

        tv2["column"] = list(selected_data)
        tv2["show"] = "headings"

        for column in tv2["column"]:
             tv2.heading(column, text=column)

        data_rows = selected_data.to_numpy().tolist()
        for row in data_rows:
            tv2.insert("", "end", values=row)
        return filled_data[selected_column]

    column_button = tk.Button(left_frame, text="Fill Missing Data", command=lambda:fill_missing())
    column_button.place(x=20, y=300)

def message_box_for_normalize():
    def normalize_all_and_close():
        filled_data=fill_all(data)
        normalize_all_data(filled_data)
        custom_dialog.destroy()
    def normalize_column_and_close():
        filled_data=fill_all(data)
        normalize_column_data(filled_data)
        custom_dialog.destroy()

    custom_dialog = tk.Toplevel()
    custom_dialog.title("Custom Dialog")
    custom_dialog.geometry("250x150")

    label = tk.Label(custom_dialog, text="What is the action you want to take?")
    label.pack()

    all_button = tk.Button(custom_dialog, text="All Data", command=lambda: normalize_all_and_close())
    all_button.place(x=10, y=30)

    column_button = tk.Button(custom_dialog, text="Only One Column", command=lambda: normalize_column_and_close())
    column_button.place(x=10, y=60)

def normalize_all_data(filled_data):

    normalized_data=normalize_all(filled_data)

    tv3.delete(*tv3.get_children())
    tv3["column"]=list(normalized_data.columns)
    tv3["show"] = "headings"

    for column in tv3["columns"]:
        tv3.heading(column, text=column)
    data_rows = normalized_data.to_numpy().tolist()

    for row in data_rows:
        tv3.insert("", "end", values=row)
        
    return normalized_data

def normalize_column_data(filled_data):

    label = tk.Label(left_frame, text="Column Name=> ")
    label.place(x=20, y=260)

    column_name = tk.Entry(left_frame)
    column_name.place(x=20, y=280)

    def normalize_one_column():
        # Kullanıcının girdiği sütun adını al
        selected_column = column_name.get()
        # Burada işlemleri gerçekleştir
        if selected_column not in filled_data.columns:
            messagebox.showwarning("Warning", f"Column '{selected_column}' not found in the dataset!")
            return
        normalized_data=normalize_column(filled_data, selected_column)
        selected_data=normalized_data[selected_column]
        tv2.delete(*tv2.get_children())

        tv2["column"] = list(selected_data)
        tv2["show"] = "headings"

        for column in tv2["column"]:
             tv2.heading(column, text=column)

        data_rows = selected_data.to_numpy().tolist()
        for row in data_rows:
            tv2.insert("", "end", values=row)
        return filled_data[selected_column]
    # İşlemi gerçekleştirecek olan buton
    column_button = tk.Button(left_frame, text="Normalize Data", command=lambda:normalize_one_column())
    column_button.place(x=20, y=300)
    return filled_data

def classification_evaulation():
    
    ## Recalls of classification steps
    filled_data=fill_missing_all(data)
    filled_data=convert_to_numeric(filled_data)
    normalized_data=normalize_all(filled_data)
    nb_classifier = CustomNaiveBayes()
    
    ## Original Data Authentic Values And Predicted Values Display
    y_test_filled, y_pred_filled = classification(filled_data)
    accuracy_filled, comparison_filled=evaluation(y_test_filled, y_pred_filled)

    label1=tk.Label(right_frame, text="Authentic Values Of Original Data:")
    label1.place(relheight=0.1, relwidth=1, rely=0)
    
    label2=tk.Label(right_frame, text=y_test_filled)
    label2.place(relheight=0.1, relwidth=1, relx=0, rely=0.06)
    
    label3=tk.Label(right_frame, text="Predicted Values Of Original Data:")
    label3.place(relheight=0.1, relwidth=1, rely=0.12)
    
    label4=tk.Label(right_frame, text=y_pred_filled)
    label4.place(relheight=0.1, relwidth=1, rely=0.18)

    label9=tk.Label(right_frame, text="Accuracy=> ")
    label9.place (relheight=0.1, relwidth=1,relx=0, rely=0.24)

    label10=tk.Label(right_frame, text=accuracy_filled)
    label10.place(relheight=0.1, relwidth=0.1, relx=0.65, rely=0.24)

    ## Normalized Data Authentic Values And Predicted Values Display
    y_test_normalized, y_pred_normalized = classification(normalized_data)
    accuracy_normalized, comparison_normalized = evaluation(y_test_normalized, y_pred_normalized)

    label5=tk.Label(right_frame, text="Authentic Values Of Normalized Data:")
    label5.place(relheight=0.1, relwidth=1, rely=0.3)
    
    label6=tk.Label(right_frame, text=y_test_normalized)
    label6.place(relheight=0.1, relwidth=1, relx=0, rely=0.36)
    
    label7=tk.Label(right_frame, text="Predicted Values Of Normalized Data:")
    label7.place(relheight=0.1, relwidth=1, rely=0.42)
    
    label8=tk.Label(right_frame, text=y_pred_normalized)
    label8.place(relheight=0.1, relwidth=1, rely=0.48)

    label11=tk.Label(right_frame, text="Accuracy=> ")
    label11.place (relheight=0.1, relwidth=1,relx=0, rely=0.54)

    label12=tk.Label(right_frame, text=accuracy_normalized)
    label12.place(relheight=0.1, relwidth=0.1, relx=0.65, rely=0.54)

    ## Comparison
    if accuracy_normalized>accuracy_filled:
        label13=tk.Label(right_frame, text="Best Accuracy is achieved with the Normalized Data Set")
        label13.place(relheight=0.1, relwidth=1, rely=0.66)
    elif accuracy_filled>accuracy_normalized:
        label13=tk.Label(right_frame, text="Best Accuracy is achieved with the Original Data Set")
        label13.place(relheight=0.1, relwidth=1, rely=0.66)
    else:
        label13=tk.Label(right_frame, text="Each Data Sets Are Equal")
        label13.place(relheight=0.1, relwidth=1, rely=0.66)

#Definitions
width = 860
height = 540 
padding_x=20
padding_y=60
root = tk.Tk()
root.geometry(f"{width}x{height}")  

## Main Frames
left_frame = tk.Frame(root, width=215, height=height)
left_frame.pack(side="left")

center_frame = tk.Frame(root, width=430, height=height)
center_frame.pack(side="left")

right_frame = tk.Frame(root, width=215, height=height)
right_frame.pack(side="left")

## Buttons
button1 = tk.Button(left_frame, text="Show Data Set", width=15, height=2, command=lambda: import_data(data,tv1))
button1.place(x=39, y=10)

button2 = tk.Button(left_frame, text="Fill Missing Values", width=15, height=2, command=lambda: message_box_for_fill(data))
button2.place(x=39, y=60)

button3 = tk.Button(left_frame, text="Normalize The Data", width=15, height=2, command=lambda: message_box_for_normalize())
button3.place(x=39, y=110)

button4 = tk.Button(left_frame, text="Classification & Evaulation", height=2, command=lambda: classification_evaulation())
button4.place(x=18, y=160)

button5 = tk.Button(left_frame, text="Exit", width=15, height=2, command=lambda: root.quit())
button5.place(x=39, y=210)

##Treewiev Widget
tv1=ttk.Treeview(center_frame)
tv1.place(relheight=0.5, relwidth=1)

tree_scroll_y = tk.Scrollbar(tv1, orient="vertical", command=tv1.yview)
tree_scroll_x = tk.Scrollbar(tv1, orient="horizontal", command=tv1.xview)

tree_scroll_x.pack(side="bottom", fill="x")
tree_scroll_y.pack(side="right", fill="y")
tv1.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)

##Treeview Widget2
tv2=ttk.Treeview(left_frame)
tv2.place(relheight=0.5, relwidth=1, rely=0.63)

tree_scroll_y = tk.Scrollbar(tv2, orient="vertical", command=tv2.yview)
tree_scroll_x = tk.Scrollbar(tv2, orient="horizontal", command=tv2.xview)

tree_scroll_x.pack(side="bottom", fill="x")
tree_scroll_y.pack(side="right", fill="y")
tv2.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)

##Treeview Widget 3
tv3=ttk.Treeview(center_frame)
tv3.place(relheight=0.5, relwidth=1, rely=0.5)

tree_scroll_y = tk.Scrollbar(tv3, orient="vertical", command=tv3.yview)
tree_scroll_x = tk.Scrollbar(tv3, orient="horizontal", command=tv3.xview)

tree_scroll_x.pack(side="bottom", fill="x")
tree_scroll_y.pack(side="right", fill="y")
tv3.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)


root.configure(bg="#333333")


root.mainloop()
