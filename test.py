from ultralytics import YOLO
import cv2
import easyocr
from cam2world_mapper import Cam2WorldMapper
import supervision as sv
import numpy as np
import time
import copy

yolo_model = YOLO('yolo11n.pt')
plate_detection_model = YOLO("yolov11_license_Plate_trained.pt")

names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

vehicles = dict()
number_plates = dict()
f = open("./test_file.txt", "+a")

# current_frame_vehicle_ids = set()
# current_frame_number_plate_ids = set()

# load video
video_path = './vehicles_e017kHOQ.mp4'

cap = cv2.VideoCapture(video_path)

reader = easyocr.Reader(['en'], gpu=True)

    # [1252, 787],
    # [2298, 803],
    # [5039, 2159],
    # [-550, 2159]


# A, B, C, D = (240, 159), (900, 100), (900, 400), (-400, 500)

A, B, C, D = (1252, 787), (2298, 803), (5039, 2159), (-550, 2159)


image_pts = [A, B, C, D]
# M6 is roughly 32 meters wide and 140 meters long there.
world_pts = [(0, 0), (24, 0), (24, 249), (0, 249)] 

mapper = Cam2WorldMapper()
mapper.find_perspective_transform(image_pts, world_pts)

def perform_ocr_on_image(img, coordinates):

    x, y, w, h = coordinates
    # Crop the plate region from the frame
    cropped_plate = img[y:y+h, x:x+w]

    # Perform OCR on the cropped plate
    result = reader.readtext(cropped_plate)

    # Extract the text (if any)
    plate_text = ""
    for res in result:
        # You can add conditions to filter valid text
        plate_text = res[1]  # Get the text from OCR result

    # print("Detected Plate Text: ", plate_text)

    return plate_text


def remove_past_number_plates():
    ## Remove unnecessary number plates
    number_plates_copy = copy.deepcopy(number_plates)
    for id in number_plates_copy:
        if id not in current_frame_number_plate_ids:
            number_plates.pop(id)


def remove_past_vehicles():
    vehicles_copy = copy.deepcopy(vehicles)
    for id in vehicles_copy:
        if id not in current_frame_vehicle_ids:
            vehicles.pop(id)


def detect_plate_to_car():
    for id in number_plates:
        plate_x, plate_y, plate_w, plate_h = number_plates[id]["x"], number_plates[id]["y"], number_plates[id]["w"], number_plates[id]["h"]
        center_x = plate_x + (plate_w - plate_x) // 2
        center_y = plate_y + (plate_h - plate_y) // 2
        for vehicle_id in vehicles:
            vehicle_x, vehicle_y, vehicle_w, vehicle_h = vehicles[vehicle_id]["x"], vehicles[vehicle_id]["y"], vehicles[vehicle_id]["w"], vehicles[vehicle_id]["h"]
            if vehicle_w > center_x > vehicle_x and vehicle_h > center_y > vehicle_y:
                print("$$$$$$$$IN$$$$$$$$$$$$$")
                if "plate_text" in number_plates[id]:
                    print("$$$$$$$$ON$$$$$$$$$$$$$")
                    vehicles[vehicle_id]["plate_number"] = number_plates[id]["plate_text"]


def detect_vehicle_speed(vehicle_results):
    for i in range(len(vehicle_results[0].boxes)):        #### Working on each bounding box element
        box_cls = vehicle_results[0].boxes.cls[i].tolist()
        if names[box_cls] == "car" or names[box_cls] == "truck" or names[box_cls] == "bus" or names[box_cls] == "motorcycle":
            box_coordinates_video = vehicle_results[0].boxes.xyxy[i].tolist()
            x, y, w, h = box_coordinates_video
            x, y, w, h = int(x), int(y), int(w), int(h)
            # print("vehicles", x, y, w, h, type(x))

            # if x >= A[0] and y >= A[1]:
            vehicle_id = vehicle_results[0].boxes.id[i]   ## Track each car with a seperate id
            box_coordinates_transformed = mapper.map(box_coordinates_video).flatten()
            x_t, y_t, w_t, h_t = box_coordinates_transformed  ## t refers to transformed
            x_t, y_t, w_t, h_t = int(x_t), int(y_t), int(w_t), int(h_t)   ## Convert to int as numpy not going to work here.
            # print(".....................")
            # Draw the bounding box and the detected text on the frame (optional)
            vehicle_id = int(vehicle_id)
            current_frame_vehicle_ids.add(vehicle_id)

            if vehicle_id not in vehicles:
                vehicles[vehicle_id] = dict()
                vehicles[vehicle_id]["x_t"] = x_t
                vehicles[vehicle_id]["y_t"] = y_t
                vehicles[vehicle_id]["w_t"] = w_t
                vehicles[vehicle_id]["h_t"] = h_t

                vehicles[vehicle_id]["x"] = x
                vehicles[vehicle_id]["y"] = y
                vehicles[vehicle_id]["w"] = w
                vehicles[vehicle_id]["h"] = h

                vehicles[vehicle_id]["last_track"] = time.time()
                vehicles[vehicle_id]["speed"] = ""
                # print("Cars: ", vehicles)
            
            else:
                # print(">>>>>>>>>>>>>>>>>>>>>>>...")
                time_diff = abs(time.time() - vehicles[vehicle_id]["last_track"])
                if time_diff >= 2:
                    vehicles[vehicle_id]["speed"] = int((abs(y_t - vehicles[vehicle_id]["y_t"]) / time_diff) * 3.6) ## Calculating the speed
                    vehicles[vehicle_id]["x_t"] = x_t
                    vehicles[vehicle_id]["y_t"] = y_t
                    vehicles[vehicle_id]["w_t"] = w_t
                    vehicles[vehicle_id]["h_t"] = h_t

                    vehicles[vehicle_id]["last_track"] = time.time()


def detect_collision(current_frame_vehicle_ids):
    collision_ids = set()
    for id_i in current_frame_vehicle_ids:
        for id_j in current_frame_vehicle_ids:
            if id_i == id_j:
                continue
            # print(vehicles[id_j]["x_t"], vehicles[id_i]["w_t"], abs(vehicles[id_i]["y_t"] - vehicles[id_j]["y_t"]))
            if vehicles[id_j]["x_t"] <= vehicles[id_i]["w_t"] and vehicles[id_j]["x_t"] >= vehicles[id_i]["x_t"] and abs(vehicles[id_i]["y_t"] - vehicles[id_j]["y_t"]) < 10 :
                collision_ids.add(id_i)
                collision_ids.add(id_j)

    return collision_ids



############################# LICENSE PLATE DETECTION ##############################

def detect_license_plate(frame):
    number_plate_result = plate_detection_model.track(frame, persist=True)

    for i in range(len(number_plate_result[0].boxes)):
        if number_plate_result[0].boxes.id != None:
            plate_id = number_plate_result[0].boxes.id[i]
            plate_id = int(plate_id)
            x, y, w, h = number_plate_result[0].boxes.xyxy[i].tolist()  # Get the bounding box coordinates
            x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
            # print(x, y, w, h)
            # plate_text = perform_ocr_on_image(frame, [x, y, w, h]) + " " + str(plate_id)
            plate_text = perform_ocr_on_image(frame, [x, y, w, h])

            current_frame_number_plate_ids.add(plate_id)
            if plate_id not in number_plates:
                number_plates[plate_id] = dict()
            number_plates[plate_id]["x"] = x
            number_plates[plate_id]["y"] = y
            number_plates[plate_id]["w"] = w
            number_plates[plate_id]["h"] = h
            number_plates["plate_text"] = plate_text
            # f.write("Number plates: " + str(number_plates))
            # # print("Number plates: ",(number_plates))

            
    #         print("Plate Text: ", plate_text)
    #         # Draw the bounding box and the detected text on the frame (optional)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # cv2.imshow('frame', frame)

    
    # # Show the frame with bounding boxes and detected text
    # cv2.imshow('frame', frame)

#############################>>>>>>>>>>>>>>>>>>>>##########################



######################## Testing on an Image ##############################

# results = model.predict("./car_plate.jpeg")
# img_path = "./car_plate2.jpeg"
# image = cv2.imread(img_path)
# # cv2.imshow("OpenCV Image",image)
# # cv2.waitKey(0)	
# print(image)
# results = model.predict(img_path)
# out = results[0].plot()

# plate = results[0].boxes.xyxy[0]
# x, y, w, h = plate  # Get the bounding box coordinates
# x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers

# # Extract the text (if any)
# plate_text = perform_ocr_on_image(image, [x, y, w, h])

#             # Draw the bounding box and the detected text on the frame (optional)
# cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
# cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# cv2.imshow('frame', image)
# cv2.waitKey(0)

# print("Detected Plate Text: ", plate_text)

###########################////////////////////////##############################

# A, B, C, D = (300, 350), (900, 350), (2700, 1200), (601, 1200)

A, B, C, D = (175, 135), (396, 139), (509, 190), (103, 191)

image_pts = [A, B, C, D]
# M6 is roughly 32 meters wide and 140 meters long there.
world_pts = [(0, 0), (20, 0), (100, 100), (0, 100)] 

mapper = Cam2WorldMapper()
mapper.find_perspective_transform(image_pts, world_pts)



ret = True
# read frames
while ret:
    ret, frame = cap.read()

    current_frame_vehicle_ids = set()
    current_frame_number_plate_ids = set()

    if ret:

        # detect objects
        # track objects
        # plate_results = plate_detection_model.track(frame, persist=True)
        # results = model.predict(frame)
        # plot results
        # cv2.rectangle
        # cv2.putText
        # print("Results:  ")
        # print(results[0].boxes)
        # print(plate_results[0].boxes)  ##Tracking each Vehicle with a seperate ID
        # print(results[0].names)        # img = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        # print(len(results[0].boxes))
        # frame_ = plate_results[0].plot()
        # print(frame)


######################## Speed Calculation Process  ############################        

        # detect_license_plate(frame) 

        vehicle_results = yolo_model.track(frame, persist=True)
        detect_vehicle_speed(vehicle_results)
        
        remove_past_number_plates()
        remove_past_vehicles()

        collision_ids = detect_collision(current_frame_vehicle_ids)
        print("collision ids: ", collision_ids)

        for id in collision_ids:
            text = str(id)
            x, y, w, h = vehicles[id]["x"], vehicles[id]["y"], vehicles[id]["w"], vehicles[id]["h"]
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("cars:", frame)


        for i in range(len(vehicle_results[0].boxes)):        #### Working on each bounding box element
            box_cls = vehicle_results[0].boxes.cls[i].tolist()
            if names[box_cls] == "car" or names[box_cls] == "truck" or names[box_cls] == "bus" or names[box_cls] == "motorcycle":
                box_coordinates_video = vehicle_results[0].boxes.xyxy[i].tolist()
                x, y, w, h = box_coordinates_video
                x, y, w, h = int(x), int(y), int(w), int(h)

                # if x >= A[0] and y >= A[1] - 50:
                vehicle_id = vehicle_results[0].boxes.id[i]   ## Track each car with a seperate id
                vehicle_id = int(vehicle_id)
                
                if len(collision_ids) != 0 and vehicle_id in collision_ids:
                    text_color = (0, 0, 255)
                else:
                    text_color = (0, 255, 0)

                # f.write( str(vehicles))
                # print(vehicles)
                # text = "id: " + str(int(car_id)) + " x: " + str(x) + " y: " + str(y) + " speed: " + str( cars[car_id]["speed"]) + " km/h"
                if "plate_number" in vehicles[vehicle_id]:
                    # print("Number plates: ", str(number_plates))

                    text = names[box_cls] + vehicles[vehicle_id]["plate_number"] + " speed: " + str( vehicles[vehicle_id]["speed"]) + " km/h"
                else:
                    text = names[box_cls] + " speed: " + str( vehicles[vehicle_id]["speed"]) + " km/h"

                cv2.rectangle(frame, (x, y), (w, h), text_color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                    # print(x, y, w, h)

        # print(car_results[0])
        cv2.imshow("cars:", frame)


####################### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #######################




######################## Testing Bounding Regions for Speed Test ###################

        # img = frame
        # # cv2.resize(img, (120, 200))
        # color1 = sv.Color.from_hex("#004080")
        # color2 = sv.Color.from_hex("#f78923")
        # # poly = np.array(((300, 350), (900, 350), (2700, 1200), (601, 1200)))  # A=1200, 700 B= 2800, 700 C = 3800, 2200 D= 501, 2200

        # poly = np.array(((240, 200), (900, 200), (900, 400), (-400, 500)))  # A=1200, 700 B= 2800, 700 C = 3800, 2200 D= 501, 2200


        # img = sv.draw_filled_polygon(img, poly, color1, 0.5)
        # img = sv.draw_polygon(img, poly, sv.Color.WHITE, 12)
        # img = sv.draw_text(img, "A", sv.Point(800, 370), color2, 2, 6)
        # img = sv.draw_text(img, "B", sv.Point(1125, 370), color2, 2, 6)  ## (100, 100), (1200, 100), (1200, 400), (-100, 400)
        # img = sv.draw_text(img, "C", sv.Point(1880, 780), color2, 2, 6)
        # img = sv.draw_text(img, "D", sv.Point(40, 780), color2, 2, 6)

        
        # cv2.imshow("check: ", img)

##################################>>>>>>>>>>################################



        # visualize
        # cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break