# knn_pose_classification_create_csv
Using MediaPipe Pose Classification soultion to get 33 landmarks of body which create and validate a training set for the k-NN classifier, and export result to a CSV. The purposeis is use it in the [ML Kit sample app](https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses#4_integrate_with_the_ml_kit_quickstart_app). 


## Folder structure:     
├───log     
├───resource     
│  ├───extract_images      
│  └───src_video     

**Video limit:**   
1.Pose type prefer counting style.   
2.Video length is around 30 seconds.   
3. Input video view only have one person.   

## Usage version 1      
*   Step 1:   
  Select the a pose video and put it in **./resource/src_video** location for prepossing. And extract the video become to images by calling the **extract_images()** function in **extract_images.py**. And run the code:      
  ```bash
		python extract_images.py   
```
*   Step 2:   
  In location **./resource/extract_images** will find a [pose_class] folder. Select what you need to classify a pose label like up-and-down which manual pick images into two folders that named **pose_class_up**, **pose_class_down**. And delete the original [pose_class] folder.   
        
		E.g., [pose_class] is a folder name 'jumping_jacks'. Manual pick up images into those folders from jumping_jacks folder by   
		creating two folders named 'jumping_jacks_up' and 'jumping_jacks_down' in './resource/extract_images/' location.   
		
		Finally, rember to delete 'jumping_jacks' folder. The structure will look like:   
		
		├───resource   
		│   ├───extract_images   
		│   │   ├───jumping_jacks_down   
		│   │   └───jumping_jacks_up     

*   Step 3:   
  Modify output path location variables in **csv_create.py** file. The variable, **export_csv** is a pose classify CSV file with KNN method result. When modify variables done, we can run the code:   
  ```bash
		python csv_create.py  
```   
## Usage version 2  
*   Run the code:   
We will see the all functions for version 1 from extract images to export CSV files with GUI. 
  ```bash
		python csv_create_GUI.py  
```   
*   Step 1:   
Input the words create a pose, and click the button 'Extract pose'. That will help you to select the source video and extract video become to images in a renamed (your input words) folder. If extract progress is made, the status will hint 'Extract done!'.   

*   Step 2-1:   
Click the 'Path select' button, and select the extract images folder. The target path will hint the complete path **./resource/extract_images**.   

*   Step 2-2:   
This step creates two status label on a pose, e.g., up-and-down. To 'Submit' button to create two states in 'pose 01' and 'pose 02', and **notice the pose state words must add '_' string in middle between poses and status**. The pose word just follow the step 1 you created.   

*   Step 2-3:   
If made the step 2-1 to step 2-2 done, we can click 'Classify' button go to the new window 'pose classify'. E.g., In step 1 create **jumping jack** pose, and step 2-2 input the **jumping jack_up** and **jumping jack_up** to submit folders. After press 'Classify' button, we will go to new window. Click **Check Status** button to refresh images display. And would be see the two button **jumping jack_up** and **jumping jack_up**.   
You can see the image to decide pose status (click up or down button) and then remember click the 'Check Status' button to refresh the image panel. Press 'Check Status' button will close the window and go back the export window when images status all you classified.      
![Pose Classify window](https://user-images.githubusercontent.com/19554347/162129252-650c976e-69a2-4020-8a06-c585ee945979.png) {:h="30%" w="30%"}
*   Step 3-1 and Step 3-2:   
Press 'Imgs log' and 'CSVs log' buttons to set path log. This steps are preparing for step 3-3 export result CSV file to debug.


## File description    

### extract_images.py   
Function describe as the below:   

**camera_info()**: To test basic camera view.

**imgs_to_video()**: Collect images which in a specific folder and convert to a video. 

**extract_images()**: Input a video(or camera) to save extract images to [pose_class] folder.

### csv_create.py   
Just modify output path location variables:   

**bootstrap_images_in_folder**: Specific folder (e.g., extract_images) need have at least a pose class or many different pose class folder.

**bootstrap_images_out_folder**: Output folders for bootstrapped images log.

**bootstrap_csvs_out_folder**: Output folders for bootstrapped CSVs log.

**export_csv**: The finally output pose label csv file we want.
		
## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install -r requirements.txt
```   
