import cv2
import os
from tqdm import tqdm
import os
import cv2
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

def extract_frames(video_path, output_dir, desired_fps):
    # 判断输出目录是否存在，如果不存在则创建，如果存在则清空目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        # 清空目录
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # 获取视频的原始帧率
    print("原始帧率：",fps)
    interval = int(round(fps / desired_fps))  # 计算每隔多少帧提取一次
    print("每隔多少帧提取一次:",interval)

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/interval)

    # 使用tqdm创建一个进度条
    progress_bar = tqdm(total=total_frames, desc='Extracting frames', unit='frames')
    total = 0
    while success:
        if count % interval == 0:
            binary_count = bin(count)[2:]  # 将count转换为二进制形式
            frame_path = os.path.join(output_dir, f"{total:04d}.jpg") # 如果要二进制，则改为binary_count
            cv2.imwrite(frame_path, image)     # 保存帧为图片
            total+=1
            progress_bar.update(1)  # 更新进度条

        success, image = vidcap.read()
        count += 1

    progress_bar.close()  # 关闭进度条

    print("Down! total:",total)


# ----------------------------------------------------------------------
        
def create_video(image_folder, output_video_path):
    # 获取文件夹中的所有图像文件
    image_files = ['tovideo/'+f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # print(image_files)
    # 按文件名排序图像，确保它们按顺序排列
    image_files.sort()
    # 创建一个ImageSequenceClip对象，它将自动将图像序列转换为视频
    clip = ImageSequenceClip(image_files, fps=30)
    # 保存视频
    clip.write_videofile(output_video_path, codec='libx264')