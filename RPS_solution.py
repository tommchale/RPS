from pickle import FRAME
import random
import cv2
from keras.models import load_model
import numpy as np
import time
import statistics


class RPS:
    def __init__(self):
        self.rps_options = ['rock', 'scissors', 'paper', 'nothing']
        self.computer_win_count = 0
        self.user_win_count = 0
        # defines shape of data array
        # an Array of 1 dimensions 224*224*3
        # numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)[source]
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Captures the video frame by frame
        # Define text details
        self.text = "Press 'c' to continue..."
        self.coordinates = (300, 300)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 2
        self.color = (255, 0, 255)
        self.thickness = 2

    def get_computer_choice(self):

        self.computer_choice = random.choice(self.rps_options[:3])

    def open_camera_with_text(self):

        # opens the camera and defines a capture object
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()

        frame = cv2.putText(frame, self.text, self.coordinates, self.font,
                            self.fontScale, self.color, self.thickness, cv2.LINE_AA)

        # Resize image
        frame = cv2.resize(frame, (480, 270))
        # shows the image from the frame read by the camera
        cv2.imshow('frame', frame)

    def open_camera_for_prediction(self):
        # opens the camera and defines a capture object
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        # resize takes cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
        # where src is the original image
        # dsize is the desired size
        # interpolation decides which pixel gets which value
        resized_frame = cv2.resize(
            frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) /
                            127.0) - 1  # Normalize the image
        self.data[0] = normalized_image
        # Resize image
        frame = cv2.resize(frame, (480, 270))
        # shows the image from the frame read by the camera
        cv2.imshow('frame', frame)

    def close_and_destroy_windows(self):
        # After the loop release the cap object
        (self.cap).release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_user_choice(self):

        timeout = time.time() + 5

        predicition_list = []

        while True:

            # Define text details
            self.text = "Press 'c' to continue..."
            self.open_camera_with_text()
            # Press c to continue the window
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        timeout = time.time() + 3

        while (timeout - time.time()) > 0:

            self.text = (
                f"{timeout - time.time():.0f}")
            self.open_camera_with_text()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.close_and_destroy_windows()

        timeout_2 = time.time() + 2

        # self.close_and_destroy_windows()

        while time.time() < timeout_2:

            self.open_camera_for_prediction()
            index = self.get_prediction()
            predicition_list.append(index)

        mode_index = statistics.mode(predicition_list)
        self.user_choice = self.rps_options[mode_index]

    def get_prediction(self):
        prediction = self.model.predict(self.data)
        index = np.argmax(prediction[0])
        return index

    def get_winner(self):

        self.get_computer_choice()
        self.get_user_choice()

        if self.user_choice == 'nothing':
            self.winner = 'none'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')

        if self.computer_choice == self.user_choice:
            self.winner = 'nothing'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('draw')

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)

        elif self.user_choice != 'nothing':
            self.winner = 'user'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)

    def play_game(self):
        # loads the tenable model
        self.model = load_model('keras_model.h5')

        while True:
            self.get_winner()
            if self.winner == 'none':
                print('Please go again and chose Rock, Paper or Scissors')
            if self.winner == 'computer':
                self.computer_win_count += 1
                print('\n The score is Computer: ',
                      self.computer_win_count, ' - User: ', self.user_win_count)
            elif self.winner == 'user':
                self.user_win_count += 1
                print('\n The score is Computer: ',
                      self.computer_win_count, ' - User: ', self.user_win_count)

            if self.computer_win_count == 1:
                print(' \n The computer has won ! \n')
                break
            elif self.user_win_count == 1:
                print(' \n The user has won ! \n')
                break

            self.close_and_destroy_windows()


def play():
    game = RPS()
    game.play_game()


play()
