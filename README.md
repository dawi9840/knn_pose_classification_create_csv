# knn_pose_classification_create_csv
Using mediapipe pose model get 33 landmarks of body which input to K-NN method do pose classification and records the result in CSV file.

## Folder structure:     
├───log     
├───resource     
│  ├───extract_images      
│  └───src_video     

Limit: The video contents must only have a single person.

## Usage   
Step1:   
  Select the a pose video and put it in **./resource/src_video** location for prepossing. And extract the video become to images by calling the **extract_images()** function in **extract_images.py**.   
  ```bash
		python extract_images.py   
```
Step2:   
  In location **./resource/extract_images** will find a [pose_class] folder. Select what you need to classify a pose label like up-and-down which manual pick images into two folders that named pose_class_up, pose_class_down. And delete the original [pose_class] folder.   

  E.g., [pose_class] is a folder name **jumping_jacks**. Manual pick up images into those folders from jumping_jacks folder by creating two folders named **jumping_jacks_up** and **jumping_jacks_down** in **./resource/extract_images/** location. **Finally, rember to delete jumping_jacks folder**. The structure will like:   
├───resource
│   ├───extract_images
│   │   ├───jumping_jacks_down
│   │   └───jumping_jacks_up     

Step3:   
  modifly output path location variables
  In **csv_create.py** file, do the pose classify with KNN method and create the result of label.csv file.


## File description    

**extract_images.py**- function describe as the below:   

		camera_info(): To test basic camera view.

		imgs_to_video(): Collect images which in a specific folder and convert to a video. 

		extract_images(): Input a video(or camera) to save extract images to [pose_class] folder.

**csv_create.py**- Just modifly output path location variables:   

		bootstrap_images_in_folder: Specific folder (e.g., extract_images) need have at least a pose class or many different pose class folder.

		bootstrap_images_out_folder: Output folders for bootstrapped images log.

		bootstrap_csvs_out_folder: Output folders for bootstrapped CSVs log.

		export_csv: The finally output pose label csv file we want.
		
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
