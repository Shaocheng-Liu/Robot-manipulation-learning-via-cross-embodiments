from PIL import Image

# only used for visualization in thesis

def gif_to_frames(gif_path, output_folder, frame_step=5):
    # Open the GIF file
    with Image.open(gif_path) as img:
        # Get the total number of frames in the GIF
        total_frames = img.n_frames

        img.seek(1)
            
        # Copy the current frame
        frame = img.copy()
        
        # Save the frame as an image
        output_path = f"{output_folder}/frame_{1}.png"
        frame.save(output_path)
        print(f"Saved: {output_path}")
        
        # Process each frame in the GIF
        for frame_num in range(0, total_frames, frame_step):
            # Seek to the frame
            img.seek(frame_num)
            
            # Copy the current frame
            frame = img.copy()
            
            # Save the frame as an image
            output_path = f"{output_folder}/frame_{frame_num}.png"
            frame.save(output_path)
            print(f"Saved: {output_path}")

# Example usage
video_file = "additional_scripts/worker_env0_success_1.0_reward_1167_sample_0.gif"  # Replace with your video file path
output_dir = "additional_scripts/extracted_frames"  # Replace with your desired output folder
frame_skip = 5  # Extract every 5th frame

# Convert the GIF
gif_to_frames(video_file, output_dir, frame_skip)
