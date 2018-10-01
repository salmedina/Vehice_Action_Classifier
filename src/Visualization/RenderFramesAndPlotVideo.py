import cv2
import os
from PIL import Image
import multiprocessing as mp
from joblib import Parallel, delayed

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), alignment='center'):
    """
    Appends images in horizontal/vertical direction.
    PARAMS:
        images: List of PIL images
        direction: 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        alignment: 'left', 'right', 'top', 'bottom', or 'center'
    RETURNS: Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if alignment == 'center':
                y = int((new_height - im.size[1])/2)
            elif alignment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if alignment == 'center':
                x = int((new_width - im.size[0])/2)
            elif alignment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

def render_video():
    video_frames_dir = '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Mixed/'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('../../output/VIRAT_S_000000_Info_tid_1.mp4', fourcc, 29.97, (1280, 720), True)
    for filename in sorted(os.listdir(video_frames_dir)):
        output_video.write(cv2.imread(os.path.join(video_frames_dir, filename)))
    output_video.release()

def combine_images(filename, frames_dir, norm_plots_dir, angle_plots_dir, save_dir, frame_size):
    frame_path = os.path.join(frames_dir, filename)
    norm_path = os.path.join(norm_plots_dir, filename)
    angle_path = os.path.join(angle_plots_dir, filename)
    save_path = os.path.join(save_dir, filename)

    plots = map(Image.open, [norm_path, angle_path])
    plots_img = append_images(plots, direction='vertical')

    mixed_img = append_images([Image.open(frame_path), plots_img], direction='horizontal', alignment='center')
    mixed_img.thumbnail(frame_size, Image.ANTIALIAS)
    w, h = mixed_img.size
    mixed_frame = Image.new('RGBA', frame_size, (0, 0, 0, 1))
    mixed_frame.paste(mixed_img, ((frame_size[0] - w) / 2, (frame_size[1] - h) / 2))
    mixed_frame.save(save_path)

def render_mixed_frames():
    frames_dir = '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Frames/'
    norm_plots_dir = '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Norms/'
    angle_plots_dir = '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Angles/'
    save_dir = '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Mixed'

    frame_size = (1280, 720)

    Parallel(n_jobs=mp.cpu_count())(delayed(combine_images)(filename=f, frames_dir=frames_dir, norm_plots_dir=norm_plots_dir, angle_plots_dir=angle_plots_dir, save_dir=save_dir, frame_size=frame_size) for f in os.listdir(norm_plots_dir))

def main():
    # print('Rendering frames')
    # render_mixed_frames()
    print('Rendering video')
    render_video()

if __name__ == '__main__':
    main()