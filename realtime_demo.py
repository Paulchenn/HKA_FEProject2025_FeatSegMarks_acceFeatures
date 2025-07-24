"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography estimation demo. Note that scene has to be planar or just rotate the camera for the estimation to work properly.
"""
# nur mit Generator

import cv2
import torch
import argparse
import threading
import numpy as np
from modules.combined_model import CombinedModel
from modules.xfeat import XFeat
#from accelerated_features.modules.xfeat import XFeat

#from modules.match_utils import draw_matches  # optional, falls du Matches anzeigen willst

# Pfad zu den Gewichten des Generators
weights_path = "/app/code"
iccv_output_wrapper = CombinedModel(weights_path)

def argparser():
    parser = argparse.ArgumentParser(description="Real-time demo using Generator and optional matching.")
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--max_kpts', type=int, default=3000)
    # parser.add_argument('--method', type=str, choices=['none', 'XFeat'], default='none',
    #                     help="Optional: apply feature matching in realtime")        # Matching-Methode
    parser.add_argument('--method', type=str, choices=['none', 'XFeat'], default='XFeat',
                        help="Optional: apply feature matching in realtime")
    parser.add_argument('--model-path', type=str, default=None,         # Pfad zum xFeat-Modell
                        help="Pfad zum trainierten XFeat-Modell (.pt oder .pth)")
    return parser.parse_args()


# Kontinuierlich das aktuelle Bild der Kamera holen
class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        #_, self.frame = self.cap.read()     # Ein erstes Bild holen
        ret, self.frame = self.cap.read()
        if not ret:
            print("Fehler beim ersten Kamerabild! Überprüfe Kamera-Index oder Zugriff.")
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
            self.frame = frame      # Aktuelles Kamerabild speichern

    def stop(self):
        self.running = False
        self.cap.release()      # Kamera freigeben

    def get_last_frame(self):
        return self.frame       # Letztes gespeichertes Bild abrufen

class GeneratorDemo:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.window_name = "Generator Output"
        self.cap = cv2.VideoCapture(args.cam)       # Kamera öffnen

        self.setup_camera()

        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()
        # Optional: XFeat für Matching laden
        # if args.method == "XFeat":
        #     self.matcher = XFeat(top_k=args.max_kpts, weights=args.model_path)
        # else:
        #     self.matcher = None

        self.matcher_enabled = args.method == "XFeat"

        # if self.matcher is not None:
        #     print("XFeat matcher initialisiert:", self.matcher)
        # else:
        #     print("Kein Matcher geladen – verwende --method XFeat")

        if self.matcher_enabled:
            print("Matching mit CombinedModel/XFeat aktiviert.")
        else:
            print("Kein Matching – verwende --method XFeat, um Feature-Matching zu aktivieren.")


        # Fenster für die Anzeige vorbereiten
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        # Fenster für gematchte Punkte vorbereiten
        cv2.namedWindow("Matched Keypoints", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Matched Keypoints", self.width * 2, self.height)
        cv2.moveWindow("Matched Keypoints", 100, 100)  # Optional: Bildschirmposition

    # Kameravorbereitung
    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    # Nachbereitung der Generator-Ausgabe    
    def preprocess(self, frame):
        img = cv2.resize(frame, (32, 32))
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # GAN-typische Normalisierung auf [-1, 1]
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img = (img - mean) / std

        return img.unsqueeze(0)

    
    def postprocess(self, tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Rück-Normalisierung von [-1, 1] auf [0, 1]
        img = (img + 1.0) / 2.0

        img = (img * 255).clip(0, 255).astype('uint8')
        return cv2.resize(img, (self.width, self.height))

    def draw_matches_window(self, img0, img1, mkpts0, mkpts1):
        # Schutz gegen leere oder ungültige Eingaben
        if img0 is None or img1 is None or mkpts0 is None or mkpts1 is None or len(mkpts0) == 0 or len(mkpts1) == 0:
            print("Nothing to draw (no keypoints).")  
            return np.zeros((self.height, self.width * 2, 3), dtype=np.uint8)

        canvas = np.hstack([img0, img1])
        offset = img0.shape[1]

        for p0, p1 in zip(mkpts0, mkpts1):
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]) + offset, int(p1[1])
            cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(canvas, (x0, y0), 2, (255, 0, 0), -1)
            cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1)

        return canvas


    def main_loop(self):
        while True:
            # frame = self.frame_grabber.get_last_frame()
            frame = self.frame_grabber.get_last_frame()
            if frame is None:
                print("WARNUNG: Frame ist None")
                continue
            if frame.sum() == 0:
                print("WARNUNG: Kamera-Frame ist schwarz (nur Nullen?)")

            cv2.imshow("RAW Kamera", frame)

            if frame is None:
                continue

            z_input = torch.randn(1, 100, 1, 1)
            # label_img = self.preprocess(frame)
            label_img = iccv_output_wrapper.preprocess_for_generator(frame)


            context_img = self.preprocess(frame)

            with torch.no_grad():
                # output = iccv_output_wrapper(z_input, label_img, context_img)
                # out_np = self.postprocess(output)
                features, out_img = iccv_output_wrapper(z_input, label_img, context_img)
                print("Generator output:", out_img.shape, out_img.min().item(), out_img.max().item())
                out_np = self.postprocess(out_img)


            # if self.matcher is not None:
            #     mkpts0, mkpts1 = self.matcher.match_xfeat(frame, out_np)

            #     # Debug-Ausgabe
            #     print(f"Matched keypoints: {len(mkpts0)} ↔ {len(mkpts1)}")

            #     # Sichere Prüfung auf gültige Matches
            #     if mkpts0 is not None and mkpts1 is not None and len(mkpts0) > 0 and len(mkpts1) > 0:
            #         match_vis = self.draw_matches_window(frame.copy(), out_np.copy(), mkpts0, mkpts1)
            #         cv2.imshow("Matched Keypoints", match_vis)

            # if self.matcher is not None:
            #     mkpts0, mkpts1 = self.matcher.match_xfeat(frame, out_np)
            if self.matcher_enabled:
                mkpts0, mkpts1, _ = iccv_output_wrapper.match_generated(frame)


                if mkpts0 is not None and mkpts1 is not None:
                    print(f"Matched keypoints: {len(mkpts0)} ↔ {len(mkpts1)}")
                else:
                    print("match_xfeat returned None")

                if mkpts0 is not None and mkpts1 is not None and len(mkpts0) > 0 and len(mkpts1) > 0:
                    match_vis = self.draw_matches_window(frame.copy(), out_np.copy(), mkpts0, mkpts1)
                    cv2.imshow("Matched Keypoints", match_vis)
                else:
                    print("Keine gültigen Matches zum Anzeigen")




            # Bild anzeigen
            cv2.imshow(self.window_name, out_np)

            # Mit q das Fenster schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = GeneratorDemo(args=argparser())
    demo.main_loop()



# import cv2
# import numpy as np
# import torch

# from time import time, sleep
# import argparse, sys, tqdm
# import threading
# from modules.combined_model import CombinedModel

# #TODO Pfad vom Kuka Rechner zu Generator etc hier einfügen
# # weights_path = "./tuned_G_119.pth" 
# # weights_path = "/app/f_e_project/tuned_G_119.pth"
# # weights_path = "/app/code/tuned_G_119.pth"
# weights_path = "/app/code"

# iccv_output_wrapper = CombinedModel(weights_path)


# from modules.xfeat import XFeat

# def argparser():
#     parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
#     parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
#     parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
#     parser.add_argument('--max_kpts', type=int, default=3_000, help='Maximum number of keypoints.')
#     parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat', help='Local feature detection method to use.')
#     parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
#     return parser.parse_args()


# class FrameGrabber(threading.Thread):
#     def __init__(self, cap):
#         super().__init__()
#         self.cap = cap
#         _, self.frame = self.cap.read()
#         self.running = False

#     def run(self):
#         self.running = True
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Can't receive frame (stream ended?).")
#             self.frame = frame
#             sleep(0.01)

#     def stop(self):
#         self.running = False
#         self.cap.release()

#     def get_last_frame(self):
#         return self.frame

# class CVWrapper():
#     def __init__(self, mtd):
#         self.mtd = mtd
#     def detectAndCompute(self, x, mask=None):
#         return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]

# class Method:
#     def __init__(self, descriptor, matcher):
#         self.descriptor = descriptor
#         self.matcher = matcher

# def init_method(method, max_kpts):
#     if method == "ORB":
#         return Method(descriptor=cv2.ORB_create(max_kpts, fastThreshold=10), matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
#     elif method == "SIFT":
#         return Method(descriptor=cv2.SIFT_create(max_kpts, contrastThreshold=-1, edgeThreshold=1000), matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True))
#     elif method == "XFeat":
#         return Method(descriptor=CVWrapper(XFeat(top_k = max_kpts)), matcher=XFeat())
#     else:
#         raise RuntimeError("Invalid Method.")


# class MatchingDemo:
#     def __init__(self, args):
#         self.args = args
#         self.cap = cv2.VideoCapture(args.cam)
#         self.width = args.width
#         self.height = args.height
#         self.ref_frame = None
#         self.ref_precomp = [[],[]]
#         self.corners = [[50, 50], [640-50, 50], [640-50, 480-50], [50, 480-50]]
#         self.current_frame = None
#         self.H = None
#         self.setup_camera()

#         #Init frame grabber thread
#         self.frame_grabber = FrameGrabber(self.cap)
#         self.frame_grabber.start()

#         #Homography params
#         self.min_inliers = 50
#         self.ransac_thr = 4.0

#         #FPS check
#         self.FPS = 0
#         self.time_list = []
#         self.max_cnt = 30 #avg FPS over this number of frames

#         #Set local feature method here -- we expect cv2 or Kornia convention
#         self.method = init_method(args.method, max_kpts=args.max_kpts)
        
#         # Setting up font for captions
#         self.font = cv2.FONT_HERSHEY_SIMPLEX
#         self.font_scale = 0.9
#         self.line_type = cv2.LINE_AA
#         self.line_color = (0,255,0)
#         self.line_thickness = 3

#         self.window_name = "Real-time matching - Press 's' to set the reference frame."

#         # Removes toolbar and status bar
#         cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
#         # Set the window size
#         cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
#         #Set Mouse Callback
#         cv2.setMouseCallback(self.window_name, self.mouse_callback)

#     def setup_camera(self):
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
#         self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
#         #self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)

#         if not self.cap.isOpened():
#             print("Cannot open camera")
#             exit()

#     def draw_quad(self, frame, point_list):
#         if len(self.corners) > 1:
#             for i in range(len(self.corners) - 1):
#                 cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
#             if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
#                 cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

#     def mouse_callback(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if len(self.corners) >= 4:
#                 self.corners = []  # Reset corners if already 4 points were clicked
#             self.corners.append((x, y))

#     def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
#         # Draw the border
#         cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
#                     color=borderColor, thickness=thickness+2, lineType=lineType)
#         # Draw the text
#         cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
#                     color=textColor, thickness=thickness, lineType=lineType)

#     def warp_points(self, points, H, x_offset = 0):
#         points_np = np.array(points, dtype='float32').reshape(-1,1,2)

#         warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
#         warped_points_np[:, 0] += x_offset
#         warped_points = warped_points_np.astype(int).tolist()
        
#         return warped_points

#     def create_top_frame(self):
#         top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
#         top_frame = np.hstack((self.ref_frame, self.current_frame))
#         color = (3, 186, 252)
#         cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
#         top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
#         # Adding captions on the top frame canvas
#         self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
#             fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

#         self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
#                     fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
#         self.draw_quad(top_frame_canvas, self.corners)
        
#         return top_frame_canvas

#     def process(self):
#         # Create a blank canvas for the top frame
#         top_frame_canvas = self.create_top_frame()

#         # Match features and draw matches on the bottom frame
#         bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)

#         # Draw warped corners
#         if self.H is not None and len(self.corners) > 1:
#             self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

#         # Stack top and bottom frames vertically on the final canvas
#         canvas = np.vstack((top_frame_canvas, bottom_frame))

#         cv2.imshow(self.window_name, canvas)

#     def match_and_draw(self, ref_frame, current_frame):

#         matches, good_matches = [], []
#         kp1, kp2 = [], []
#         points1, points2 = [], []

#         # Detect and compute features
#         if self.args.method in ['SIFT', 'ORB']:
#             kp1, des1 = self.ref_precomp
#             kp2, des2 = self.method.descriptor.detectAndCompute(current_frame, None)
#         else:
#             current = self.method.descriptor.detectAndCompute(current_frame)
#             kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
#             kpts2, descs2 = current['keypoints'], current['descriptors']
#             idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
#             points1 = kpts1[idx0].cpu().numpy()
#             points2 = kpts2[idx1].cpu().numpy()

#         if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
#             # Match descriptors
#             matches = self.method.matcher.match(des1, des2)

#             if len(matches) > 10:
#                 points1 = np.zeros((len(matches), 2), dtype=np.float32)
#                 points2 = np.zeros((len(matches), 2), dtype=np.float32)

#                 for i, match in enumerate(matches):
#                     points1[i, :] = kp1[match.queryIdx].pt
#                     points2[i, :] = kp2[match.trainIdx].pt

#         if len(points1) > 10 and len(points2) > 10:
#             # Find homography
#             self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
#             inliers = inliers.flatten() > 0

#             if inliers.sum() < self.min_inliers:
#                 self.H = None

#             if self.args.method in ["SIFT", "ORB"]:
#                 good_matches = [m for i,m in enumerate(matches) if inliers[i]]
#             else:
#                 kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
#                 kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
#                 good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

#             # Draw matches
#             matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
#         else:
#             matched_frame = np.hstack([ref_frame, current_frame])

#         color = (240, 89, 169)

#         # Add a colored rectangle to separate from the top frame
#         cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

#         # Adding captions on the top frame canvas
#         self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
#             fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
#                 # Adding captions on the top frame canvas
#         self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
#             fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

#         return matched_frame

#     def main_loop(self):
#         self.current_frame = self.frame_grabber.get_last_frame()
#         self.ref_frame = self.current_frame.copy()
#         self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

#         while True:
#             if self.current_frame is None:
#                 break

#             t0 = time()
#             self.process()

#             key = cv2.waitKey(1)
#             if key == ord('q'):
#                 break
#             elif key == ord('s'):
#                 self.ref_frame = self.current_frame.copy()  # Update reference frame
#                 self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

#             self.current_frame = self.frame_grabber.get_last_frame()

#             #Measure avg. FPS
#             self.time_list.append(time()-t0)
#             if len(self.time_list) > self.max_cnt:
#                 self.time_list.pop(0)
#             self.FPS = 1.0 / np.array(self.time_list).mean()
        
#         self.cleanup()

#     def cleanup(self):
#         self.frame_grabber.stop()
#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     demo = MatchingDemo(args = argparser())
#     demo.main_loop()
