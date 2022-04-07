import os
import shutil
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from pose_embedding import FullBodyPoseEmbedder
from pose_classification import PoseClassifier
from bootstrap_helper import BootstrapHelper
from extract_images import extract_images
from csv_create import dump_for_the_app

def main():
    window = tk.Tk()
    window.title('Pose classify export CSV')
    window.resizable(width=0, height=0) # (x,y): (1,1) window could be resizable.
    window.geometry('1280x720')
    cursors_lst =[
        'arrow', 'circle', 'clock', 'cross', 'dotbox', 'exchange', 
        'fleur', 'heart', 'man', 'mouse', 'pirate', 'plus', 
        'shuttle', 'sizing', 'spider', 'spraycan', 'star', 
        'target', 'tcross', 'trek'
    ]
    window.config(cursor=cursors_lst[0])

    def del_window():
        window.destroy()

    def get_select_folder_name():
        # https://stackoverflow.com/questions/28373288/get-file-path-from-askopenfilename-function-in-tkinter
        file = fd.askopenfile()
        if file: 
            return file.name

    def select_path():   
        path_ = fd.askdirectory()
        path.set(path_)
 
    def create_file():  
        # print(f'folder_name1: {folder.get()}')
        # print(f'folder_name2: {folder2.get()}')
        # print(f'path_name: {path.get()}')
        dirs = os.path.join(path.get(), folder.get())
        dirs2 = os.path.join(path.get(), folder2.get())

        if not os.path.exists(dirs):
            os.makedirs(dirs)
            tkinter.messagebox.showinfo('Tips:', folder.get() + ' created successfully!') 
        else:
            tkinter.messagebox.showerror('Tips','Pose 01 path is exists or not empty.')

        if not os.path.exists(dirs2):
            os.makedirs(dirs2)
            tkinter.messagebox.showinfo('Tips:', folder2.get() + ' created successfully!')
        else:
            tkinter.messagebox.showerror('Tips','Pose 02 path is exists or not empty.')

    def get_extract_images():
        # Get text input string.
        result = step1_input_text.get("1.0","end")
        tk.Label(window,text=result, width=50, bg='green', fg='yellow', font=(font_type, font_size))
        text_result = result.replace('\n', '')

        if len(text_result) > 1:
            # Get select folder name.
            folder_name = get_select_folder_name()
            # Invoke extract_images function extract images.
            extract_images(cap_index=str(folder_name), pose_class=str(text_result))
            extract_lab = tk.Label(window, text='Extract Done!', bg='green', fg='yellow', font=(font_type, font_size))
            extract_lab.grid(column=3, row=2)
        else:
            print('Please create pose classify string')
            tkinter.messagebox.showerror('Tips','Pose classific name must to create!')

    def select_imgs_log_folder():
        knn_out_imgs_log_ = fd.askdirectory()
        knn_out_imgs_log.set(knn_out_imgs_log_)

    def select_csvs_log_folder():
        knn_out_csvs_log_ = fd.askdirectory()
        knn_out_csvs_log.set(knn_out_csvs_log_)

    def set_export_csv():
        result_step3_3 = step3_3_input_text.get("1.0","end")
        text_result_step3_3 = result_step3_3.replace('\n', '')

        if text_result_step3_3[-4:len(text_result_step3_3)] == '.csv':
            export_csv_path_ = fd.askdirectory()
            export_csv_path.set(export_csv_path_)
            step3_3_path = str(export_csv_path.get())

            if len(step3_3_path) > 2:
                export_result_file = step3_3_path + '/' + text_result_step3_3
                # print(f'export_result_file: {export_result_file}')

                export_CSV(
                    bootstrap_images_in_folder=str(path.get()), 
                    bootstrap_images_out_folder=str(knn_out_imgs_log.get()), 
                    bootstrap_csvs_out_folder=str(knn_out_csvs_log.get()), 
                    export_csv=export_result_file)

                export_status = 'Done! location: ' + export_result_file
                step3_4_lab_b = tk.Label(window, text=export_status, bg='purple', fg='yellow', font=(font_type, font_size))
                step3_4_lab_b.grid(column=3, row=12)
            else:
                print('Please assign export file path.')
                tkinter.messagebox.showerror('Tips','Please assign export file path.')  
        else:
            print('Make sure export file format is ".csv".')
            tkinter.messagebox.showerror('Tips','Make sure export file format is ".csv".')

    def export_CSV(
        bootstrap_images_in_folder, 
        bootstrap_images_out_folder, 
        bootstrap_csvs_out_folder, 
        export_csv):
        if (len(bootstrap_images_in_folder) > 2 and 
                len(bootstrap_images_out_folder) > 2 and 
                    len(bootstrap_csvs_out_folder)> 2 and 
                        len(export_csv) > 2):
            # Initialize helper.
            bootstrap_helper = BootstrapHelper(
                images_in_folder=bootstrap_images_in_folder,
                images_out_folder=bootstrap_images_out_folder,
                csvs_out_folder=bootstrap_csvs_out_folder)
            # Check how many pose classes and images for them are available.
            print('\nCheck how many pose classes and images for them are available:')
            bootstrap_helper.print_images_in_statistics()
            
            # Bootstrap all images.
            # Set limit to some small number for debug.
            bootstrap_helper.bootstrap()

            # Check how many images were bootstrapped.
            print('\nCheck how many images were bootstrapped:')
            bootstrap_helper.print_images_out_statistics()

            # After initial bootstrapping images without detected poses were still saved
            # in the folder (but not in the CSVs) for debug purpose. Let's remove them.
            print('\nAlign CSVs with filtered images:')
            bootstrap_helper.align_images_and_csvs(print_removed_items=False)
            bootstrap_helper.print_images_out_statistics()
            
            # Automatic filtrationBootstrapHelper
            # Transforms pose landmarks into embedding.
            pose_embedder = FullBodyPoseEmbedder()

            # Classifies give pose against database of poses.
            pose_classifier = PoseClassifier(
                pose_samples_folder=bootstrap_csvs_out_folder,
                pose_embedder=pose_embedder,
                top_n_by_max_distance=30,
                top_n_by_mean_distance=10)

            outliers = pose_classifier.find_pose_sample_outliers()
            print('Number of outliers: ', len(outliers))

            # Analyze outliers.
            bootstrap_helper.analyze_outliers(outliers)

            # Remove all outliers (if you don't want to manually pick).
            bootstrap_helper.remove_outliers(outliers)

            # Align CSVs with images after removing outliers.
            bootstrap_helper.align_images_and_csvs(print_removed_items=False)
            bootstrap_helper.print_images_out_statistics()
            
            # Dump for the App
            dump_for_the_app(pose_samples_folder=bootstrap_csvs_out_folder, pose_samples_csv_path=export_csv)
        else:
            tkinter.messagebox.showerror('Tips','Please make sure step2-1 or step3-1 to step3-3 is not empty.')

    def get_imgs_list(img_path:str):       
        # Get file names from directory.
        # img_path = 'C:/Users/user/Desktop/knn_pose_classification_create_csv/resource/extract_images/Jab_Cross_down'
        # img_path = str(path.get()) + '/' + pose_name
        file_list = os.listdir(img_path)
        # print (f'file_list: {file_list[0]}')

        if not file_list :
            print("Directory is empty")
            return 0
        else:
            return file_list

    def get_same_str(str1:str, str2:str):
        # Get the same string from str1 and str2.
        set01, set02 = set(str1.split('_')), set(str2.split('_'))
        same_dict = set01 & set02
        same_str = ''.join(same_dict)
        # print(f'input1: {str1}, type:{type(str1)}\ninput2: {str2}, type:{type(str2)}')
        # print(f'\nset01: {set01}\nset02: {set02}')
        # print(f'\nsame_str: {same_str}, type: {type(same_str)}')
        return same_str

    def open_new_window():
        extract_imgs_path = str(path.get())
        pose01 = str(folder.get())
        pose02 = str(folder2.get())
        pose01_path = extract_imgs_path + '/' + pose01
        pose02_path = extract_imgs_path + '/' + pose02
        control_mode = 0

        def move_image(src_file, dest_file):
            # Move src_file image to dest_file location.
            shutil.move(src_file, dest_file)
            # print(f'\nsrc_file: {src_file}')
            # print(f'dest_file: {dest_file}')
            print('Move1 done!')

        def btn_show_img(img_panel):
            # img_path = extract_imgs_path + '/' + choose_pose + '/' + img_file_name[2]
            bbbbb = 'C:/Users/user/Desktop/knn_pose_classification_create_csv/002.jpg'
            print(f'\npose2: {bbbbb}')

            p2_img = Image.open(bbbbb)
            p2_img = p2_img.resize((350, 350), Image.ANTIALIAS)
            p2_img = ImageTk.PhotoImage(p2_img)

            img_panel.configure(image=p2_img)
            img_panel.image = p2_img
    
        def check_status():
            # './resource/extract_images/Jab_Cross_down/Jab_Cross052.jpg'
            img_path = str(path.get()) + '/' + choose_pose
            img_file_name = get_imgs_list(img_path)

            if img_file_name == 0:
                new_lab1 = tk.Label(new_window, text=choose_pose, width=t_w1, font=(font_type, font_size))
                new_lab2 = tk.Label(new_window, text='Folder image is empty.', font=(font_type, font_size))
                shutil.rmtree(img_path, ignore_errors=True)
                print('Remove original folder done.')
                
                # Image label layout.
                new_lab1.place(x=400, y=300)
                new_lab2.place(x=400, y=340)

                # Delete new_window.
                new_window.destroy()
            else:
                img_path = extract_imgs_path + '/' + choose_pose + '/' + img_file_name[0]

                # the enumerate object
                enumerate_object = enumerate(img_file_name)
                # first iteration from enumerate(enumerate_object)
                iteration = next(enumerate_object) 
                index, item = iteration
                # print(f'item: {item}')

                src_file = extract_imgs_path + '/' + choose_pose + '/' + item
                dest_file1 = pose01_path + '/' + item
                dest_file2 = pose02_path + '/' + item

                # Image label components.
                new_lab1 = tk.Label(new_window, text=choose_pose, width=t_w1, font=(font_type, font_size))
                # new_lab2 = tk.Label(new_window, text=img_path, font=(font_type, font_size))
                new_lab2 = tk.Label(new_window, text=item, font=(font_type, font_size))

                p2_img = Image.open(img_path)
                p2_img = p2_img.resize((350, 350), Image.ANTIALIAS)
                p2_img = ImageTk.PhotoImage(p2_img)

                img_panel = tk.Label(new_window, image=p2_img)
                img_panel.image = p2_img

                # Button components.
                new_btn2 = tk.Button(
                    new_window, text=pose01, 
                    width=t_w3+2, height=btn_h_size+1, 
                    bg='red', fg='yellow', 
                    font=(font_type, font_size), 
                    command=lambda: move_image(src_file=src_file, dest_file=dest_file1))

                new_btn3 = tk.Button(
                    new_window, text=pose02, 
                    width=t_w3+2, height=btn_h_size+1, 
                    bg='blue', fg='yellow', 
                    font=(font_type, font_size), 
                    command=lambda: move_image(src_file=src_file, dest_file=dest_file2))

                # Image label layout.
                new_lab1.place(x=400, y=300)
                new_lab2.place(x=400, y=340)
                img_panel.place(x=10, y=0)

                # Button layout.
                new_btn2.place(x=15, y=360)
                new_btn3.place(x=195, y=360)

        if os.path.exists(pose01_path) and len(pose01) > 1:
            control_mode = 1
            # print(f'\npose01_path: {pose01_path}')
        else:
            tkinter.messagebox.showerror('Tips','Check step 2-2 folder path is correct.')

        if os.path.exists(pose02_path) and len(pose02) > 1:
            control_mode = 1
            # print(f'pose02_path: {pose02_path}')
        else:
            tkinter.messagebox.showerror('Tips','Check step 2-2 folder path is correct.')

        if (os.path.exists(pose01_path) and len(pose01) > 1 and
                os.path.exists(pose02_path) and len(pose02) > 1 and 
                    control_mode == 1):
            control_mode = 2
            # TODO : choose_pose, debug mode to set.
            # Choose pose: Get the same string from pose01 and pose02.
            choose_pose = get_same_str(str1=pose01, str2=pose02)

        if os.path.exists(extract_imgs_path):
            if control_mode == 2:
                new_window = tk.Toplevel(window)
                new_window.title('Pose classify')
                new_window.geometry('720x640')
                new_window.resizable(width=0, height=0)
                new_window.config(cursor=cursors_lst[11])
                
                new_btn0 = tk.Button(
                    new_window, text='Check Status', width=t_w3+20, 
                    height=btn_h_size+1, bg='grey', fg='white', 
                    font=(font_type, font_size), 
                    command=check_status)
                new_btn0.place(x=15, y=435)

            else:
                tkinter.messagebox.showerror('Tips','Please check step 2-2.')
        else:
            print('Please assign step2-1 path.')
            tkinter.messagebox.showerror('Tips','Please assign step 2-1 path.')
            print(f'extract_imgs_path: {extract_imgs_path}')

    # Global variables ---------------------------------------------------------
    t_w1, t_w2, t_w3 = 10, 15, 13
    font_size, btn_h_size = 15, 1
    entry_width = 107
    font_type = 'Calibri'

    # Receiving user's file_path selection.
    path = tk.StringVar()   

    # Receiving user's folder_name selection.
    folder = tk.StringVar() 
    folder2 = tk.StringVar()

    # Receiving export variables.
    knn_out_imgs_log = tk.StringVar()
    knn_out_csvs_log = tk.StringVar()
    export_csv_path = tk.StringVar()

    # window GUI components ---------------------------------------------------------
    step1_lab = tk.Label(window, text='step 1:', width=t_w1, font=(font_type, font_size))
    step1_lab_a = tk.Label(window,text='Class create:', width=t_w2)
    step1_btn = tk.Button(window, text='Extract pose', width=t_w3, height=btn_h_size, bg='green', fg='yellow',  font=(font_type, font_size), command=get_extract_images)
    step1_input_text = tk.Text(window, width=entry_width, height=1)
    step1_2_lab_a = tk.Label(window,text='Status:', width=t_w2)

    space_lab0 = tk.Label(window, text='  ', font=(font_type, font_size))
    space_lab1 = tk.Label(window, text='  ', font=(font_type, font_size))

    step2_1_lab = tk.Label(window, text='step 2-1:', width=t_w1, font=(font_type, font_size))
    step2_1_lab_a = tk.Label(window,text='Target path:', width=t_w2)
    step2_1_btn = tk.Button(window, text='Path select', width=t_w3, height=btn_h_size, bg='blue', fg='yellow', font=(font_type, font_size), command=select_path)
    step2_1_entry = tk.Entry(window, width=entry_width, textvariable=path)

    step2_2_lab = tk.Label(window, text='step 2-2:', width=t_w1, font=(font_type, font_size))
    step2_2_lab_a = tk.Label(window,text='Pose 01:', width=t_w2)
    step2_2_btn = tk.Button(window, text='Submit', width=t_w3, height=btn_h_size, bg='blue', fg='yellow', font=(font_type, font_size), command=create_file)
    step2_2_entry = tk.Entry(window, width=entry_width, textvariable=folder)

    step2_2_b_lab_a = tk.Label(window,text='Pose 02:', width=t_w2)
    step2_2_b_entry = tk.Entry(window, width=entry_width, textvariable=folder2)

    step2_3_lab = tk.Label(window, text='step 2-3:', width=t_w1, font=(font_type, font_size))
    step2_3_btn = tk.Button(window, text='Classify', width=t_w3, height=btn_h_size, bg='blue', fg='yellow', font=(font_type, font_size), command=open_new_window)

    space_lab2 = tk.Label(window, text='  ', font=(font_type, font_size))

    step3_1_lab = tk.Label(window, text='step 3-1:', width=t_w1, font=(font_type, font_size))
    step3_1_lab_a = tk.Label(window,text='Set path:', width=t_w2)
    step3_1_btn = tk.Button(window, text='Imgs log', width=t_w3, height=btn_h_size, bg='purple', fg='yellow', font=(font_type, font_size), command=select_imgs_log_folder)
    step3_1_entry = tk.Entry(window, width=entry_width, textvariable=knn_out_imgs_log)

    step3_2_lab = tk.Label(window, text='step 3-2:', width=t_w1, font=(font_type, font_size))
    step3_2_lab_a = tk.Label(window,text='Set path:', width=t_w2)
    step3_2_btn = tk.Button(window, text='CSVs log', width=t_w3, height=btn_h_size, bg='purple', fg='yellow', font=(font_type, font_size), command=select_csvs_log_folder)
    step3_2_entry = tk.Entry(window, width=entry_width, textvariable=knn_out_csvs_log)

    step3_3_lab = tk.Label(window, text='step 3-3:', width=t_w1, font=(font_type, font_size))
    step3_3_lab_a = tk.Label(window,text='Set file:', width=t_w2)
    step3_3_btn = tk.Button(window, text='  Export CSV ',  width=t_w3, height=btn_h_size, bg='purple', fg='yellow', font=(font_type, font_size), command=set_export_csv)
    step3_3_input_text =tk.Text(window, width=entry_width, height=1)
    step3_4_lab_a = tk.Label(window,text='Status:', width=t_w2)

    space_lab3 = tk.Label(window, text='  ', font=(font_type, font_size))
    exit_btn = tk.Button(window, text='Exit',  width=t_w3, height=btn_h_size, bg='pink', fg='red', font=(font_type, font_size), command=del_window)

    # window GUI layout ---------------------------------------------------------
    position_column, position_row = 0, 1
    step1_lab.grid(column=position_column, row=position_row)
    step1_lab_a.grid(column=position_column+2, row=position_row)
    step1_btn.grid(column=position_column+1, row=position_row)
    step1_input_text.grid(column=position_column+3, row=position_row, columnspan=30)
    step1_2_lab_a.grid(column=position_column+2, row=position_row+1)

    space_lab0.grid(column=position_column, row=position_row+1)
    space_lab1.grid(column=position_column, row=position_row+2)

    step2_1_lab.grid(column=0, row=position_row+3)
    step2_1_lab_a.grid(column=position_column+2, row=position_row+3)
    step2_1_btn.grid(column=position_column+1, row=position_row+3)
    step2_1_entry.grid(column=position_column+3, row=position_row+3, columnspan=30)

    step2_2_lab.grid(column=position_column, row=position_row+4)
    step2_2_lab_a.grid(column=position_column+2, row=position_row+4)
    step2_2_btn.grid(column=position_column+1, row=position_row+4)
    step2_2_entry.grid(column=position_column+3, row=position_row+4, columnspan=30)

    step2_2_b_lab_a.grid(column=position_column+2, row=position_row+5)
    step2_2_b_entry.grid(column=position_column+3, row=position_row+5, columnspan=30)

    step2_3_lab.grid(column=position_column, row=position_row+6)
    step2_3_btn.grid(column=position_column+1, row=position_row+6)

    space_lab2.grid(column=position_column, row=position_row+7)

    step3_1_lab.grid(column=position_column, row=position_row+8)
    step3_1_lab_a.grid(column=position_column+2, row=position_row+8)
    step3_1_btn.grid(column=position_column+1, row=position_row+8)
    step3_1_entry.grid(column=position_column+3, row=position_row+8, columnspan=30)

    step3_2_lab.grid(column=position_column, row=position_row+9)
    step3_2_lab_a.grid(column=position_column+2, row=position_row+9)
    step3_2_btn.grid(column=position_column+1, row=position_row+9)
    step3_2_entry.grid(column=position_column+3, row=position_row+9, columnspan=30)

    step3_3_lab.grid(column=position_column, row=position_row+10)
    step3_3_lab_a.grid(column=position_column+2, row=position_row+10)
    step3_3_btn.grid(column=position_column+1, row=position_row+10)
    step3_3_input_text.grid(column=position_column+3, row=position_row+10, columnspan=30)
    step3_4_lab_a.grid(column=position_column+2, row=position_row+11)

    space_lab3.grid(column=position_column, row=position_row+12)
    exit_btn.grid(column=position_column+1, row=position_row+13)

    window.mainloop()

if __name__ == '__main__':

    main()