import cv2
import os

def frame_preprocess(path, undistort=False, intr=None, dist_intr=None, dist=None):
    stream = cv2.VideoCapture(path)
    assert stream.isOpened(), 'Cannot capture source'

    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_imgs = []
    im_names = []
    
    frame_num = 0
    for k in range(datalen):
        if k % 3 == 0 or k % 3 == 1 or k % 3 == 2:
            (grabbed, frame) = stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                stream.release()
                break

            # orig_imgs.append(frame[:, :, ::-1])
            if undistort:
                frame = cv2.undistort(frame, intr, dist, None, dist_intr)
            orig_imgs.append(frame)
            im_names.append(f'{frame_num:08d}' + '.jpg')
            frame_num += 1
    H, W, _ = frame.shape
    stream.release()

    # print(f'Total number of frames: {frame_num} in {path}')
    return im_names, orig_imgs, H, W

def create_video_writer(filename, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def add_text_to_frame(frame, text, position=(60, 60), font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=1, font_color=(0, 0, 0), thickness=2):
    # Add text on the frame
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
    return frame

def convert_video_ffmpeg(input_path):
    temp_path = input_path + '.temp'
    output_path = input_path
    
    # Rename the original file
    os.rename(input_path, temp_path)
    
    # Construct the ffmpeg command with the -y option
    ffmpeg_command = f'ffmpeg -i {temp_path} -vcodec libx264 -y {output_path}'
    
    # Execute the command
    os.system(ffmpeg_command)
    
    # Remove the temporary file
    os.remove(temp_path)