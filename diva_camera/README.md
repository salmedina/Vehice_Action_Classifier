Usage Documentation
====
```sh
usage: scene_classifier.py [-h] [--video_dir VIDEO_DIR]
                 [--video_lst_file VIDEO_LST_FILE] [--out_dir OUT_DIR]

functions: classify the camera id for the video, ? for unseen camera

optional arguments:
  -h, --help            show this help message and exit
  --video_dir VIDEO_DIR
                        the root directory path of cropped video proposals
                        (default: None) (default: None)
  --video_lst_file VIDEO_LST_FILE
                        the path of video list, in this file each line is the
                        relative path of the video to the video_dir. That is,
                        video_file_path = os.path.join(video_dir, ${line})
                        (default: None) Note ${line} may contain "/" (default:
                        None)
  --out_dir OUT_DIR     the root directory of outputs: the camera id of each
                        video is stored in the corresponding file
                        "${out_dir}/${line}.camera. (default: None)
```

RUN
===
1. Build the image
```sh
nvidia-docker build -t hima:scene .
```

2. test in interactive shell
```sh
nvidia-docker run --rm -it -v /data:/tmp hima:scene /bin/bash
python app/scene_classifier.py --video_dir /tmp/video --video_lst_file /tmp/video.lst  --out_dir .
```