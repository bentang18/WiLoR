1. preprocesss
2. 2d keypoint estimation
3. triangulate for 3d keypoints
4. fit mano model
5. temporal processing - butterworth or other

For this project, I am creating a 3D hand pose estimation pipeline for patients with dystonia, a movement disorder characterized by irregular joint positions and abnormal movements of the hand. For this project, I wish to recover precise biomechanical data to quantify dystonia severity.

The input is as follows: I have RGB video (typically around three minutes) from three angles, front, left, and right of the patient playing piano, banjo, or other behavioral assays. For around half of the videos I also have explicit checkboard calibration (around 1-2 minutes in length, multiple angles and rotations of the checkboard for all views).

The desired output is as follows: an estimated 3d MANO model per frame, with consistent B parameters for each video. A visualization (could be heuristic e.g. abnormal) of dystonia severity is also desired: this could be in the form of highlighting parts of the mano model where the dystonia is manifested.

The pipeline consists mainly of five steps: 

1. Preprocessing: videos are first split into individual frames. Then a video restoration model is run on the images (RVRT) for motion deblurring. Then the deblurred images are fed into a hand box detector (WiLoR currently) which creates crops of the hands. Then sharpening methods are used on the hand box crops: CLAHE + Mild Unsharp Mask.
2. 2d keypoint estimation: These preprocessed hand box crops are fed into WiLoR. WiLoR will produce 3d joint keypoints. Reproject these 3d keypoints back onto the 2d image to obtain 2d joint keypoints. Future considerations is replacing this step with sapiens, however, the first MVP will use WiLoR.
3. triangulate for 3d keypoints: first we need to recover camera intrinsics and extrinsics. For 
4. fit mano model

Create a checklist of the components that are already implemented and components that need to be implemented. I will check the checklist for accuracy.

Future considerations:
- is sapiens better than WiLoR for 2d hand pose estimation?