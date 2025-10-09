import os
import cv2
import glob
import numpy as np
# import ft2
from PIL import ImageFont, ImageDraw, Image
from pdf2image import convert_from_path

fontpath = "/Users/ayshrv/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Avenir-Medium.ttf"
font = ImageFont.truetype(fontpath, 36)

def combine_videos_grid2x3(thermal_paths, kaist_paths, output_path, frame_size=(1064//2, 1114//2), fps=15):
    """
    Combine 3 thermal_im and 3 kaist videos into a single video in a 2x3 grid with more horizontal than vertical padding.
    Top row: [T1, T2, K1], Bottom row: [T3, K3, K2]
    For each frame t, if t%2==0 add 'RGB' below every frame, else add 'Thermal' below every frame.
    """
    assert len(thermal_paths) == 3 and len(kaist_paths) == 3, "Need 3 videos from each source."
    
    # Open all video captures
    caps = [cv2.VideoCapture(p) for p in thermal_paths + kaist_paths]
    
    # Get min frame count
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    min_frames = min(frame_counts)
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    
    text_band_height = 50
    h_padding = 100  # horizontal padding between columns
    v_padding = 20  # vertical padding between rows
    frame_w, frame_h = frame_size[0], frame_size[1] + text_band_height
    # 3 frames + 2 h_paddings per row, 2 rows + 1 v_padding between rows
    grid_width = frame_w * 3 + h_padding * 2
    grid_height = frame_h * 2 + v_padding
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    for t in range(min_frames):
        frames = []
        if t>=0 and t<fps:
            label = "RGB"
        elif t>=2*fps and t<3*fps:
            label = "Thermal"
        elif t>=4*fps and t<5*fps:
            label = "RGB"
        else:
            label = ""
        
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            else:
                # trim frame
                frame = frame[:-50, :]
                frame = cv2.resize(frame, frame_size)
            # Add text band below
            canvas = np.ones((frame_size[1] + text_band_height, frame_size[0], 3), dtype=np.uint8) * 255
            canvas[:frame_size[1], :] = frame
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (frame_size[0] - text_width) // 2
            text_y = frame_size[1] + (text_band_height - text_height) // 2
            # Create a transparent overlay for the text
            overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.text((text_x, text_y), label, font=font, fill=(0, 0, 0, 128))  # 128 = 50% opacity

            # Composite the overlay onto the original image
            img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
            frames.append(np.array(img_pil))
        # Top row: [T1, T2, K1] = frames[0], frames[1], frames[3]
        # Bottom row: [T3, K3, K2] = frames[2], frames[5], frames[4]
        def row_with_padding(indices):
            row_frames = [frames[i] for i in indices]
            padded_row = row_frames[0]
            for f in row_frames[1:]:
                vpad = np.ones((frame_h, h_padding, 3), dtype=np.uint8) * 255
                padded_row = np.hstack([padded_row, vpad, f])
            return padded_row
        top_row = row_with_padding([0, 1, 3])
        bottom_row = row_with_padding([2, 5, 4])
        hpad = np.ones((v_padding, grid_width, 3), dtype=np.uint8) * 255
        grid = np.vstack([top_row, hpad, bottom_row])
        out.write(grid)
    
    # Release everything
    for cap in caps:
        cap.release()
    out.release()
    print(f"Saved grid video to {output_path}")


def get_opacity(t, t_low, t_high):
    return int(255 * (t - t_low) / (t_high - t_low))

def combine_videos_grid1x3(video_paths, output_path, frame_size=(1166//2, 936//2), fps=15):
    """
    Combine 3 videos into a single video in a 1x3 grid with horizontal padding between columns.
    For each frame t, if t%2==0 add 'RGB' below every frame, else add 'Thermal' below every frame.
    """
    assert len(video_paths) == 3, "Need 3 videos."

    # Open all video captures
    caps = [cv2.VideoCapture(p) for p in video_paths]

    # Get min frame count
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    min_frames = min(frame_counts)
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    text_band_height = 50
    h_padding = 100  # horizontal padding between columns
    frame_w, frame_h = frame_size[0], frame_size[1] + text_band_height
    # 3 frames + 2 h_paddings per row
    grid_width = frame_w * 3 + h_padding * 2
    grid_height = frame_h
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))

    for t in range(min_frames):
        if t>=0 and t<2*fps:
            label = "RGB"
            opacity = 255
        elif t>=2*fps and t<2*fps+fps//2:
            label = "Depth"
            opacity = get_opacity(t, 2*fps, 2*fps+fps//2)
        elif t>=2*fps+fps//2 and t<4*fps+fps//2:
            label = "Depth"
            opacity = 255
        elif t>=4*fps+fps//2 and t<4*fps+2*fps//2:
            label = "RGB"
            opacity = get_opacity(t, 4*fps+fps//2, 4*fps+2*fps//2)
        elif t>=4*fps+2*fps//2 and t<6*fps+2*fps//2:
            label = "RGB"
            opacity = 255
        else:
            label = ""
            opacity = 0
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            else:
                # trim frame
                frame = frame[:-50, :]
                frame = cv2.resize(frame, frame_size)
            # Add text band below
            canvas = np.ones((frame_size[1] + text_band_height, frame_size[0], 3), dtype=np.uint8) * 255
            canvas[:frame_size[1], :] = frame
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (frame_size[0] - text_width) // 2
            text_y = frame_size[1] + (text_band_height - text_height) // 2 - 10
            # Create a transparent overlay for the text
            overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.text((text_x, text_y), label, font=font, fill=(0, 0, 0, opacity))  # 128 = 50% opacity

            # Composite the overlay onto the original image
            img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
            frames.append(np.array(img_pil))
        # Concatenate frames horizontally with padding
        padded_row = frames[0]
        for f in frames[1:]:
            vpad = np.ones((frame_h, h_padding, 3), dtype=np.uint8) * 255
            padded_row = np.hstack([padded_row, vpad, f])
        out.write(padded_row)

    # Release everything
    for cap in caps:
        cap.release()
    out.release()
    print(f"Saved 1x3 grid video to {output_path}")

diff_texts = [
    "RGB", "Depth"
]

nyu_depth_data_root = "nyu_depth"

# read all videos in the folder
nyu_depth_video_paths = glob.glob(os.path.join(nyu_depth_data_root, '*.mp4'))

# sort
nyu_depth_video_paths = sorted(nyu_depth_video_paths)

# # now change the order to be [,4,8,12], [1,5,9,13]
# kaist_video_paths = kaist_video_paths[::3] + kaist_video_paths[1::3] \
#     + kaist_video_paths[2::3]
# thermal_im_video_paths = thermal_im_video_paths[::3] + thermal_im_video_paths[1::3] \
#     + thermal_im_video_paths[2::3]


# fps = 15
# batch_size = 3
# num_batches = min(len(thermal_im_video_paths), len(kaist_video_paths)) // batch_size

# for i in range(num_batches):
#     thermal_batch = thermal_im_video_paths[i*batch_size:(i+1)*batch_size]
#     kaist_batch = kaist_video_paths[i*batch_size:(i+1)*batch_size]
#     output_path = f"rgb_thermal_grid_batch_{i+1}.mp4"
#     combine_videos_grid1x3(thermal_batch, kaist_batch, output_path, fps=15)


fps = 15
batch_size = 3
num_batches = len(nyu_depth_video_paths) // batch_size

# Chunk thermal_im_video_paths and kaist_video_paths into batches of 3 videos alternating between them
video_paths = []
for i in range(num_batches):
    video_paths.append(nyu_depth_video_paths[i*batch_size:(i+1)*batch_size])

for i in range(num_batches):
    output_path = f"rgb_depth_grid_1x3_{i+1}.mp4"
    combine_videos_grid1x3(video_paths[i], output_path, fps=15)


