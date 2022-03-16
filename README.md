# knn_pose_classification_create_csv
Using mediapipe pose model get 33 landmarks of body which input to K-NN method do pose classification and records the result in CSV file.

## Folder structure:     
├───log     
├───resource     
│  ├───extract_images      
│  └───src_video     

Limit: The video contents must only have a single person.

## Usage   
Step1: Select the pose label and  put the video in ./resource/src_video folder location for prepossing. 

Step2: Extract the video become to images and define the label folder name by calling the extract_images() function in extract_images.py.

Step3: In csv_create.py file, do the pose classify with KNN method and create the result of label.csv file.


## File description    

**extract_images.py**:    
Function describe:   

		camera_info(): To test basic camera view.

		imgs_to_video(): Collect images which in a specific folder and convert to a video. 

		extract_images(): Input a video(or camera) to save extract images to [pose_class] folder.

**csv_create.py**:   
Just modifly output path location variables:   

		[bootstrap_images_in_folder]: Specific folder (e.g., extract_images) need have at least a pose class or many different pose class folder.

		[bootstrap_images_out_folder]: Output folders for bootstrapped images log.

		[bootstrap_csvs_out_folder]: Output folders for bootstrapped CSVs log.

		[export_csv]: The finally output pose label csv file we want.
		
## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install numpy==1.19.3
pip install opencv-python==4.5.1.48
pip install tqdm==4.56.0
pip install pillow==8.1.0
pip install matplotlib==3.3.4
pip install requests==2.25.1
pip install mediapipe==0.8.3
```   
