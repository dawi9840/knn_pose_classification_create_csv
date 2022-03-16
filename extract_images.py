import os
import cv2
import glob


def camera_info(cap_index, set_cap_w=None, set_cap_h=None, save_output=None):
    cap=cv2.VideoCapture(cap_index)

    if ((set_cap_w == None) and (set_cap_h == None)):
        pass
    else:
        if ((set_cap_w <= 0) or (set_cap_h <= 0)):
            pass
        else:
            # set_cap_w:int, set_cap_h:int
            cap.set(3, set_cap_w)
            cap.set(4, set_cap_h)

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_w, cap_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')
        print(f'w: {cap_w}, h: {cap_h}')

    if save_output == None:
        pass
    else:
        output_fps = input_fps - 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 輸出附檔名為 mp4
        out = cv2.VideoWriter(save_output, fourcc, output_fps, (cap_w, cap_h))

    while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                if (save_output == None):
                    pass
                else:
                    # Determine to save the video.
                    out.write(frame)
                    # image2 = cv2.flip(image, 1)
                    # out.write(image2)

                # cv2.imshow('Raw Video Feed', frame)
                cv2.imshow('Raw flip Video Feed ', cv2.flip(frame, 1))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def extract_images(cap_index, pose_class:str):
    '''Input a video(or camera) to save extract images to [pose_class] folder.'''
    cap = cv2.VideoCapture(cap_index)

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    count = 0
    while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                pose_class_path = './resource/extract_images/' + pose_class
                # Check folder is exist or not.
                isdir = os.path.isdir(pose_class_path) 
                
                if isdir == True:
                    # print(f'file path is exist? : {isdir}') 

                    # Save frame as JPEG file.
                    cv2.imwrite(pose_class_path + '/'+pose_class+'{0:0>3}.jpg'.format(count), frame)
                    print(f'Frame: {count}, Location:{pose_class_path}')
                    count += 1
                    pass
                else:
                    # Create a new floder.
                    pose_class_path = os.mkdir(pose_class_path)
                    print('Create a floder!')

                cv2.imshow('Extract video view', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    print('Extract images done!')
    cap.release()
    cv2.destroyAllWindows()


def imgs_to_video(input_imgs_floder:str, output_video:str):
    ''''
    input_imgs_floder = './*.jpg', 
    output_video = file name.
    '''
    img_array = []

    for filename in glob.glob(input_imgs_floder):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print('create done!')


def main():
    img_path = [
        './resource/src_image/img_knee_to_chest.png',
    ]
    video_path = [
        0, # your camera number
        './resource/src_video/Jab_Cross_01.mp4',
    ]

    # Example usage:
    # camera_info(cap_index=video_path[0], set_cap_w=1280, set_cap_h=720, save_output=None)
    imgs_to_video(input_imgs_floder='./resource/src_image/*.jpg', output_video='imgs_merge_to_video.mp4')
    # extract_images(cap_index=video_path[1], pose_class='Jab_Cross_01')


if __name__ == '__main__':

    main()